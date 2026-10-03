#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import logging
from itertools import product
from types import SimpleNamespace
from unittest import main, TestCase

import torch
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList
from torchbiggraph.stats import Stats
from torchbiggraph.train_cpu import IterationManager
from torchbiggraph.train_unified import UnifiedTrainingCoordinator


class TestIterationManager(TestCase):
    def test_full(self) -> None:
        im = IterationManager(
            num_epochs=2, edge_paths=["A", "B", "C"], num_edge_chunks=4
        )
        self.assertEqual(list(im), list(product(range(2), range(3), range(4))))

    def test_partial(self) -> None:
        im = IterationManager(
            num_epochs=2,
            edge_paths=["A", "B", "C"],
            num_edge_chunks=4,
            iteration_idx=(1 * 3 + 2) * 4 + 3,
        )
        self.assertEqual(list(im), [(1, 2, 3)])

    def test_tampering(self) -> None:
        im = IterationManager(
            num_epochs=2, edge_paths=["A", "B", "C"], num_edge_chunks=4
        )
        it = iter(im)
        self.assertEqual(next(it), (0, 0, 0))
        im.iteration_idx = (0 * 3 + 1) * 4 + 1
        # When calling next it gets incremented.
        self.assertEqual(next(it), (0, 1, 2))
        im.edge_paths = ["foo", "bar"]
        im.num_edge_chunks = 2
        self.assertEqual(next(it), (1, 1, 1))
        im.iteration_idx = 100
        with self.assertRaises(StopIteration):
            next(it)

    def test_properties(self) -> None:
        im = IterationManager(
            num_epochs=2,
            edge_paths=["A", "B", "C"],
            num_edge_chunks=4,
            iteration_idx=(0 * 3 + 1) * 4 + 2,
        )
        self.assertEqual(im.epoch_idx, 0)
        self.assertEqual(im.edge_path_idx, 1)
        self.assertEqual(im.edge_path, "B")
        self.assertEqual(im.edge_chunk_idx, 2)
        self.assertEqual(
            im.get_checkpoint_metadata(),
            {
                "iteration/num_epochs": 2,
                "iteration/epoch_idx": 0,
                "iteration/num_edge_paths": 3,
                "iteration/edge_path_idx": 1,
                "iteration/edge_path": "B",
                "iteration/num_edge_chunks": 4,
                "iteration/edge_chunk_idx": 2,
            },
        )


class TestUnifiedCoordinateTrain(TestCase):
    def _run(self, num_dynamic_rels: int) -> None:
        num_edges, num_rels = 1000, 3
        lhs = torch.arange(num_edges)
        rhs = torch.arange(num_edges) + num_edges
        rel = torch.randint(num_rels, (num_edges,))
        edges = EdgeList(
            EntityList.from_tensor(lhs.clone()),
            EntityList.from_tensor(rhs.clone()),
            rel.clone(),
        )
        eval_edge_idxs = torch.randint(num_edges, (50,))

        batches = []

        class RecordingTrainer:
            def process_one_batch(self, model, batch):
                # Batches are views of the edge tensors, which are modified
                # again after training, so take a snapshot.
                batches.append(
                    EdgeList(
                        EntityList.from_tensor(batch.lhs.tensor.clone()),
                        EntityList.from_tensor(batch.rhs.tensor.clone()),
                        batch.rel.clone(),
                    )
                )
                return Stats(count=len(batch))

        coordinator = UnifiedTrainingCoordinator.__new__(UnifiedTrainingCoordinator)
        coordinator.device = torch.device("cpu")
        coordinator.bucket_logger = logging.getLogger("test")
        coordinator.model = SimpleNamespace(num_dynamic_rels=num_dynamic_rels)
        coordinator.trainer = RecordingTrainer()
        coordinator.config = SimpleNamespace(batch_size=64)
        coordinator._coordinate_train(edges, eval_edge_idxs, epoch_idx=0)

        # Every training edge is seen exactly once, and no eval edge is.
        trained = torch.cat([b.lhs.tensor for b in batches])
        eval_set = set(eval_edge_idxs.tolist())
        expected = sorted(set(range(num_edges)) - eval_set)
        self.assertEqual(sorted(trained.tolist()), expected)
        for b in batches:
            # Edges kept their own rhs and relation type through the reordering.
            self.assertTrue(torch.equal(b.rhs.tensor, b.lhs.tensor + num_edges))
            self.assertTrue(torch.equal(b.rel.expand(len(b)), rel[b.lhs.tensor]))
            if num_dynamic_rels == 0:
                self.assertTrue(b.has_scalar_relation_type())
        # Eval edges are back at their original positions for evaluation.
        self.assertTrue(torch.equal(edges.lhs.tensor[eval_edge_idxs], eval_edge_idxs))
        self.assertTrue(
            torch.equal(edges.rel[eval_edge_idxs], rel[eval_edge_idxs])
        )

    def test_grouped_by_relation_type(self) -> None:
        self._run(num_dynamic_rels=0)

    def test_dynamic_relations(self) -> None:
        self._run(num_dynamic_rels=3)


if __name__ == "__main__":
    main()
