FAQ & Troubleshooting
=====================

Frequently Asked Questions
--------------------------

Undirected graphs
^^^^^^^^^^^^^^^^^

Edges in PBG's :ref:`data model <data-model>` are always interpreted as directed.
To operate on undirected data, it is often enough to replace each undirected edge
```a <-> b`` with two directed edges ``a -> b`` and ``b -> a``. In fact, even with data
that is already directed, it may be beneficial to artificially add a "reversed"
edge ``b ~> a`` for each original edge ``a -> b``, of a different relation type.
This is automatically done by PBG in the :ref:`dynamic relations mode <dynamic-relations>`.

Training on a GPU with unified memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The regular GPU trainer (``num_gpus``) is designed for GPUs with their own
memory: it keeps the embeddings of the active partitions in CPU shared memory
and copies them to the GPU for training, and it also keeps two CPU copies of
the current chunk of edges plus one on the GPU. On machines where the CPU and
GPU share the same physical memory, such as an NVIDIA DGX Spark, all these
copies take up the same RAM.

Setting ``unified_memory=True`` trains on a single GPU while keeping a single
copy of everything as CUDA tensors. Roughly, for each bucket this needs:

- ``4 * D`` bytes per entity of the active partitions for the embeddings, plus
  4 bytes for the optimizer state. The buffers for the largest partitions are
  allocated at startup, so running out of memory happens immediately.
- 24 bytes per edge of the current edge chunk, plus up to 24 bytes per edge
  while the chunk is being shuffled. The size of a chunk is controlled by
  ``num_edge_chunks`` or ``max_edges_per_chunk``.

All entity types must have the same number of partitions, and featurized
entities, edge weights and distributed training aren't supported in this mode.
No worker processes are created, so ``workers`` has no effect. It may help to
set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``, so that edge chunks of
different sizes don't fragment GPU memory.

Common issues
-------------

Bus error
^^^^^^^^^

Training might occasionally fail with a ``Bus error (core dumped)`` message, and
no traceback. This is often caused by the inability to allocate enough _shared_
memory, that is, memory that can be simultaneously accessed by multiple processes
(this is needed to perform training in parallel). Such an error is produced by
the kernel and it's hard to detect it in advance or catch it.

This may occur when running PBG inside a Docker container, as by default the
shared memory limit for them is rather small. `This PyTorch issue <https://github.com/pytorch/pytorch/issues/2244>`_
may provide some insight in how to address that. If this occurs on a Linux machine,
it may be fixed by increasing the size of the ``tmpfs`` mount on ``/dev/shm`` or
on ``/var/run/shm``.

It may also just be that the machine ran out of physical memory because the data
is too large. This is exactly the scenario PBG was designed for, and can be fixed
by increasing the number of partitions of the data.
