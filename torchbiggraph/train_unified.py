#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

"""Single-GPU training for machines whose GPU shares memory with the CPU.

The regular GPU trainer (train_gpu.py) is designed for discrete GPUs: it keeps
every active partition in CPU shared memory and copies it to the GPU for each
sub-bucket, so the embeddings, the optimizer state and the edges all exist
twice (the edges three times). On a unified-memory machine such as a DGX Spark,
both copies live in the same physical RAM, so that only wastes memory.

This coordinator keeps a single copy of everything, as CUDA tensors, in the
main process:

- embedding buffers are allocated on the GPU up front (one or two per entity
  type, as in the CPU trainer) and checkpoints are read/written through a small
  CPU staging buffer;
- edges are read from disk straight into GPU tensors, through the same kind of
  staging buffer;
- there is a single sub-bucket per bucket, so the entity shuffle that the GPU
  trainer uses to build random sub-buckets is not needed;
- training and evaluation run in-process, so no worker processes are created.
"""

import errno
import logging
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Set

import h5py
import numpy as np
import torch
from torchbiggraph.batching import AbstractBatchProcessor, process_in_batches
from torchbiggraph.config import ConfigSchema
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList
from torchbiggraph.graph_storages import (
    AbstractEdgeStorage,
    FileEdgeStorage,
    FORMAT_VERSION,
    FORMAT_VERSION_ATTR,
)
from torchbiggraph.model import MultiRelationEmbedder
from torchbiggraph.stats import Stats, StatsHandler
from torchbiggraph.tensorlist import TensorList
from torchbiggraph.train_cpu import NOOP_STATS_HANDLER, TrainingCoordinator
from torchbiggraph.types import (
    Bucket,
    EntityName,
    LongTensorType,
    Rank,
    SINGLE_TRAINER,
)
from torchbiggraph.util import (
    CouldNotLoadData,
    div_roundup,
    EmbeddingHolder,
    round_up_to_nearest_multiple,
)


logger = logging.getLogger("torchbiggraph")

# Edges are read from disk into a CPU staging buffer of this many bytes per
# column (lhs, rhs, rel) and then copied to the GPU.
EDGE_STAGING_BUFFER_BYTES = 256 * 1024 * 1024


class UnifiedTrainingCoordinator(TrainingCoordinator):
    def __init__(
        self,
        config: ConfigSchema,
        model: Optional[MultiRelationEmbedder] = None,
        trainer: Optional[AbstractBatchProcessor] = None,
        evaluator: Optional[AbstractBatchProcessor] = None,
        rank: Rank = SINGLE_TRAINER,
        subprocess_init: Optional[Callable[[], None]] = None,
        stats_handler: StatsHandler = NOOP_STATS_HANDLER,
        device: Optional[torch.device] = None,
    ):
        if config.num_machines > 1 or config.num_partition_servers > 0:
            raise RuntimeError("unified_memory only supports a single machine")
        if config.num_gpus > 1:
            raise RuntimeError("unified_memory only supports a single GPU")
        for entity_name, entity in config.entities.items():
            if entity.featurized:
                raise RuntimeError(
                    f"unified_memory doesn't support featurized entities "
                    f"({entity_name})"
                )
        if config.half_precision:
            for entity in config.entities:
                # need this for tensor cores to work
                assert config.entity_dimension(entity) % 8 == 0
            assert config.batch_size % 8 == 0
            assert config.num_batch_negs % 8 == 0
            assert config.num_uniform_negs % 8 == 0

        # Needs to be set before calling the superclass, which allocates the
        # embedding buffers through _make_embedding_storage_freelist.
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("unified_memory requires a CUDA device")
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type == "cuda":
            torch.cuda.set_device(device)
        self.device = device

        super().__init__(
            config, model, trainer, evaluator, rank, subprocess_init, stats_handler
        )

    # override
    def _make_embedding_storage_freelist(
        self, entity_counts: Dict[str, List[int]], holder: EmbeddingHolder
    ) -> Dict[EntityName, Set[torch.FloatStorage]]:
        if holder.lhs_unpartitioned_types or holder.rhs_unpartitioned_types:
            raise RuntimeError(
                "unified_memory requires all entity types to have the same number "
                "of partitions"
            )
        # Same sizing as the CPU trainer: one buffer per side an entity type
        # appears on, big enough for its largest partition. They're allocated up
        # front so that GPU memory usage is fixed and an OOM happens at startup.
        embedding_storage_freelist: Dict[EntityName, Set[torch.FloatStorage]] = (
            defaultdict(set)
        )
        total_bytes = 0
        for entity_type, counts in entity_counts.items():
            max_count = max(counts)
            if holder.nparts_lhs == 1 and holder.nparts_rhs == 1:
                num_sides = 1
            else:
                num_sides = (
                    1 if entity_type in holder.lhs_partitioned_types else 0
                ) + (1 if entity_type in holder.rhs_partitioned_types else 0)
            dimension = self.config.entity_dimension(entity_type)
            for _ in range(num_sides):
                buffer = torch.empty(
                    (max_count * dimension,), dtype=torch.float, device=self.device
                )
                embedding_storage_freelist[entity_type].add(buffer.storage())
                total_bytes += buffer.numel() * buffer.element_size()
        logger.info(
            f"Allocated {total_bytes:,} bytes of embedding buffers on {self.device}"
        )
        return embedding_storage_freelist

    # override
    def _create_worker_pool(self, rank, subprocess_init):
        # Training and evaluation both happen in this process, on the GPU.
        return None

    # override
    def _load_edges(
        self,
        edge_storage: AbstractEdgeStorage,
        bucket: Bucket,
        edge_chunk_idx: int,
        num_edge_chunks: int,
    ) -> EdgeList:
        if isinstance(edge_storage, FileEdgeStorage):
            return self._load_file_edges(
                edge_storage, bucket, edge_chunk_idx, num_edge_chunks
            )
        # Other storages can only load into CPU memory: keep that copy only
        # until it has been moved to the GPU.
        edges = edge_storage.load_chunk_of_edges(
            bucket.lhs, bucket.rhs, edge_chunk_idx, num_edge_chunks, shared=False
        )
        if edges.weight is not None:
            raise NotImplementedError("Edge weights not implemented in GPU mode yet")
        return EdgeList(
            self._entity_list(edges.lhs.tensor.to(self.device)),
            self._entity_list(edges.rhs.tensor.to(self.device)),
            edges.rel.to(self.device),
        )

    def _load_file_edges(
        self,
        edge_storage: FileEdgeStorage,
        bucket: Bucket,
        edge_chunk_idx: int,
        num_edge_chunks: int,
    ) -> EdgeList:
        # Mirrors FileEdgeStorage.load_chunk_of_edges, but reads into GPU
        # tensors through a fixed-size staging buffer.
        file_path = edge_storage.get_edges_file(bucket.lhs, bucket.rhs)
        try:
            hf = h5py.File(file_path, "r")
        except OSError as err:
            # Same as FileEdgeStorage: report missing files as CouldNotLoadData.
            if f"errno = {errno.ENOENT}" in str(err):
                raise CouldNotLoadData() from err
            raise err
        with hf:
            if hf.attrs.get(FORMAT_VERSION_ATTR, None) != FORMAT_VERSION:
                raise RuntimeError(f"Version mismatch in edge file {file_path}")
            if "weight" in hf:
                raise NotImplementedError(
                    "Edge weights not implemented in GPU mode yet"
                )
            for key in ("lhsd_data", "rhsd_data"):
                if key in hf and hf[key].len() > 0:
                    raise NotImplementedError(
                        "Featurized entities not implemented in unified_memory mode"
                    )
            num_edges = hf["rel"].len()
            chunk_size = div_roundup(num_edges, num_edge_chunks)
            begin = edge_chunk_idx * chunk_size
            end = min((edge_chunk_idx + 1) * chunk_size, num_edges)
            lhs = self._read_column(hf["lhs"], begin, end)
            rhs = self._read_column(hf["rhs"], begin, end)
            rel = self._read_column(hf["rel"], begin, end)
        return EdgeList(self._entity_list(lhs), self._entity_list(rhs), rel)

    def _read_column(
        self, dataset: h5py.Dataset, begin: int, end: int
    ) -> LongTensorType:
        out = torch.empty((max(end - begin, 0),), dtype=torch.long, device=self.device)
        step = max(1, EDGE_STAGING_BUFFER_BYTES // out.element_size())
        staging = torch.empty((min(step, out.numel()),), dtype=torch.long)
        for offset in range(0, out.numel(), step):
            n = min(step, out.numel() - offset)
            dataset.read_direct(
                staging.numpy(),
                source_sel=np.s_[begin + offset : begin + offset + n],
                dest_sel=np.s_[:n],
            )
            out[offset : offset + n].copy_(staging[:n])
        return out

    def _entity_list(self, tensor: LongTensorType) -> EntityList:
        # EntityList.from_tensor creates its (empty) TensorList on the CPU, as a
        # zero-stride view; moving that to the GPU would materialize 8 bytes per
        # edge. Build the empty TensorList directly on the device instead.
        offsets = torch.zeros((), dtype=torch.long, device=tensor.device).expand(
            (tensor.shape[0] + 1,)
        )
        data = torch.empty((0,), dtype=torch.long, device=tensor.device)
        return EntityList(tensor, TensorList(offsets, data))

    # override
    def _coordinate_train(self, edges, eval_edge_idxs, epoch_idx) -> Stats:
        assert edges.weight is None, "Edge weights not implemented in GPU mode yet"
        bucket_logger = self.bucket_logger
        columns = [edges.lhs.tensor, edges.rhs.tensor]
        if edges.rel.dim() > 0:
            columns.append(edges.rel)
        num_edges = len(edges)

        # The edge tensors are reordered in place, to avoid making a second
        # copy of them: the training edges are moved to the front and shuffled.
        # The eval edges are stashed and written back afterwards, so that the
        # evaluation after training sees them at eval_edge_idxs, as expected.
        if eval_edge_idxs is not None:
            bucket_logger.debug("Removing eval edges")
            eval_edge_idxs = eval_edge_idxs.to(self.device)
            eval_values = [c[eval_edge_idxs] for c in columns]
            is_train = torch.ones((num_edges,), dtype=torch.bool, device=self.device)
            is_train[eval_edge_idxs] = False
            num_train_edges = int(is_train.sum())
            for c in columns:
                c[:num_train_edges] = c[is_train]
            del is_train
        else:
            num_train_edges = num_edges

        bucket_logger.debug("Shuffling edges")
        perm = torch.randperm(num_train_edges, device=self.device)
        self._permute_in_place(columns, num_train_edges, perm)
        del perm

        lhs = edges.lhs.tensor[:num_train_edges]
        rhs = edges.rhs.tensor[:num_train_edges]
        if self.model.num_dynamic_rels > 0 or edges.rel.dim() == 0:
            # Batches may mix relation types, so they're just slices of the
            # edges and process_in_batches doesn't copy anything.
            train_edges = EdgeList(
                self._entity_list(lhs),
                self._entity_list(rhs),
                edges.rel[:num_train_edges] if edges.rel.dim() > 0 else edges.rel,
            )
            stats = process_in_batches(
                batch_size=self.config.batch_size,
                model=self.model,
                batch_processor=self.trainer,
                edges=train_edges,
            )
        else:
            # Each batch must have a single relation type. Instead of
            # batching.group_by_relation_type, which makes a sorted copy of all
            # the edges, sort them in place. The sort is stable, so the edges of
            # each relation type stay shuffled.
            rel = edges.rel[:num_train_edges]
            order = torch.sort(rel, stable=True).indices
            self._permute_in_place(columns, num_train_edges, order)
            del order
            rel_types, counts = torch.unique_consecutive(rel, return_counts=True)
            groups: List[EdgeList] = []
            offset = 0
            for rel_type, count in zip(rel_types, counts.tolist()):
                groups.append(
                    EdgeList(
                        self._entity_list(lhs[offset : offset + count]),
                        self._entity_list(rhs[offset : offset + count]),
                        rel_type,
                    )
                )
                offset += count
            stats = Stats.sum(
                [
                    self.trainer.process_one_batch(self.model, batch)
                    for batch in _batch_groups(groups, self.config.batch_size)
                ]
            )

        if eval_edge_idxs is not None:
            bucket_logger.debug("Restoring eval edges")
            for c, v in zip(columns, eval_values):
                c[eval_edge_idxs] = v

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return stats.average()

    @staticmethod
    def _permute_in_place(
        columns: List[LongTensorType], n: int, perm: LongTensorType
    ) -> None:
        # Only one column's worth of temporary memory is needed at a time.
        for c in columns:
            c[:n] = c[:n][perm]

    # override
    def _coordinate_eval(self, edges, eval_edge_idxs) -> Optional[Stats]:
        if eval_edge_idxs is None:
            return None
        eval_batch_size = round_up_to_nearest_multiple(
            self.config.batch_size, self.config.eval_num_batch_negs
        )
        self.bucket_logger.debug("Performing evaluation")
        # Gather the eval edges here rather than passing indices to
        # process_in_batches, whose TensorList indexing creates CPU tensors.
        eval_edge_idxs = eval_edge_idxs.to(self.device)
        eval_edges = EdgeList(
            self._entity_list(edges.lhs.tensor[eval_edge_idxs]),
            self._entity_list(edges.rhs.tensor[eval_edge_idxs]),
            edges.rel[eval_edge_idxs] if edges.rel.dim() > 0 else edges.rel,
        )
        stats = process_in_batches(
            batch_size=eval_batch_size,
            model=self.model,
            batch_processor=self.evaluator,
            edges=eval_edges,
        )
        return stats.average()


def _batch_groups(groups: List[EdgeList], batch_size: int) -> Iterable[EdgeList]:
    """Same as batching.batch_edges_group_by_relation_type, on pre-split groups

    Picks the group of the next batch at random, weighted by how many edges each
    group has left, so that relation types are interleaved during training.
    """
    num_edges_left_per_group = torch.tensor(
        [len(group) for group in groups], dtype=torch.long
    )
    while num_edges_left_per_group.sum() > 0:
        idx = int(torch.multinomial(num_edges_left_per_group.float(), 1))
        group = groups[idx]
        offset = len(group) - int(num_edges_left_per_group[idx])
        batch = group[offset : offset + batch_size]
        yield batch
        num_edges_left_per_group[idx] -= len(batch)
