# Handoff: unified-memory training on DGX Spark

Context for whoever picks this up on the Spark (human or Claude). Branch:
`unified-memory-coordinator`.

## The problem

Training PBG on a DGX Spark (128 GB unified CPU/GPU memory), 256-dim
embeddings. A graph 1/5 the size of the current one used ~90% of memory at
batch_size 50k; the current graph OOMed at the same settings. Dropping
batch_size to 10k didn't help much, because batch size barely affects memory.
The current run uses ~89 GB total while `nvidia-smi` reports ~39 GB.

## Where the memory goes in the existing GPU trainer (`train_gpu.py`)

`nvidia-smi` only counts CUDA allocations. PBG's main copy of the data lives
in `/dev/shm` and is pinned with `cudaHostRegister`, which `nvidia-smi` doesn't
count, but on the Spark it's the same physical RAM. Per bucket:

- **Embeddings, twice:** shared-memory buffers (`train_cpu.py`,
  `_make_embedding_storage_freelist`) plus a full GPU copy in `do_one_job`.
  With 1 GPU there's a single sub-partition, so the GPU copy is the whole
  partition.
- **Edges, three times:** the loaded chunk (24 B/edge, `/dev/shm`), the
  sub-bucket buffers (another 24 B/edge, sized for the largest bucket chunk),
  and the GPU copy (24 B/edge).
- **The `workers` pool:** one process per CPU core. It is NOT idle in GPU
  mode: `_coordinate_eval` isn't overridden by the GPU trainer, so evaluation
  (`eval_fraction` defaults to 0.05) runs there on the CPU.
- **Default `max_edges_per_chunk`:** 1e9, which can mean ~72 GB of edge
  buffers. Set `num_edge_chunks` explicitly.

## Design decision

Options considered:

1. **Zero-copy views of the pinned `/dev/shm` buffers.** Smallest patch, but
   GPU speed is unknown: small OS pages may cause GPU TLB misses on random
   embedding lookups.
2. **`cudaMallocManaged`.** Can't be shared across processes, and PyTorch has
   no "both CPU and CUDA" tensor.
3. **Chosen: everything as ordinary CUDA tensors, single process.** Moving the
   CPU-only work (checkpoint I/O, shuffle) off the CPU makes this possible.
   GPU speed is the same as today's GPU-side training by construction, and it
   needs no C++ extension.

## What was implemented (`torchbiggraph/train_unified.py`)

Enabled with `unified_memory=True` in the config; `train.py` selects
`UnifiedTrainingCoordinator`.

- Embedding buffers are allocated on the GPU at startup, so an OOM happens
  immediately.
- Checkpoints are read and written through a 256 MB CPU staging buffer
  (`checkpoint_storage.py`).
- Edges are read from HDF5 straight into GPU tensors through a staging buffer.
- No entity shuffle or sub-bucketing (one sub-bucket with 1 GPU).
- Eval edges are moved aside, then training edges are shuffled and stably
  sorted by relation type in place. This avoids the full sorted copy that
  `batching.group_by_relation_type` makes. Eval edges are restored afterwards.
- Training and eval run in-process, and no worker pool is created.
- Base-class hooks added in `train_cpu.py`: `_make_embedding_storage_freelist`,
  `_create_worker_pool`, `_load_edges`. The previous bucket's edges are now
  freed before the next bucket loads.
- Optimizer state is always serialized as CPU tensors
  (`checkpoint_manager.py`).

Expected memory: ~(4·D + 4) B per active entity, plus 24 B/edge in the
current chunk, plus up to 24 B/edge transiently while shuffling.

Limits: 1 GPU, 1 machine, all entity types with the same partition count, no
featurized entities, no edge weights.

## Status

- Tested on a Mac **without CUDA** by running the coordinator with
  `device=cpu`. Four functional tests plus two unit tests pass. The full suite
  shows no regressions (6 failures pre-exist on `main` on macOS: batching
  order and distributed tests).
- **Never run on a GPU.** Four GPU tests in `test/test_functional.py`
  (`test_unified_gpu*`) skip without CUDA.

## Next steps on the Spark

1. Check out the branch and install it: `pip install -e .`
2. Run `python -m pytest test -k "unified or Unified"`. This is the first real
   GPU run, so fix anything device-related.
3. Do a short real run with `unified_memory=True` and an explicit
   `num_edge_chunks`. Check that `nvidia-smi` and total system memory now
   roughly agree.
4. Optionally set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce
   fragmentation from different-sized edge chunks.
5. Compare throughput (edges/sec in the logs) against the old GPU trainer.

## Open items

- **Possible race in `train_gpu.py`** (untouched): the GPU→CPU copy-back uses
  `non_blocking=True` with no `torch.cuda.synchronize()` before the worker
  reports done. The main process then immediately runs the reverse shuffle on
  those buffers.
- **Not tried yet:** making the shared-memory pages larger (huge pages for
  `/dev/shm`) for the old trainer.
