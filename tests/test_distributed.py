import os
import unittest
from unittest.mock import patch

import torch

from lm_from_zero.training import (
    DistributedContext,
    DistributedError,
    distributed_session,
)


class DistributedContextTests(unittest.TestCase):
    def test_single_process_fallback_and_topology_validation(self) -> None:
        context = DistributedContext.current()
        self.assertFalse(context.enabled)
        self.assertTrue(context.is_primary)
        self.assertEqual(context.torch_device("cpu"), torch.device("cpu"))
        self.assertEqual(
            context.reduce_float(
                3.0,
                reduction="mean",
                device=torch.device("cpu"),
            ),
            3.0,
        )
        tensor = torch.tensor(4.0)
        self.assertEqual(float(context.reduce_tensor_mean(tensor)), 4.0)
        self.assertEqual(context.all_gather_object("state"), ("state",))
        self.assertEqual(context.broadcast_primary_object("control"), "control")
        context.validate_topology(rank=0, world_size=1)
        with self.assertRaisesRegex(DistributedError, "process group"):
            context.validate_topology(rank=1, world_size=2)

    def test_collectives_reduce_and_exchange_rank_ordered_values(self) -> None:
        context = DistributedContext(
            rank=1,
            world_size=2,
            local_rank=1,
            backend="gloo",
        )

        def double(tensor: torch.Tensor, *, op: object) -> None:
            del op
            tensor.mul_(2)

        def gather(output: list[object], value: object) -> None:
            output[:] = ["rank-zero", value]

        def broadcast(payload: list[object], *, src: int) -> None:
            self.assertEqual(src, 0)
            payload[0] = "from-primary"

        with (
            patch("torch.distributed.all_reduce", side_effect=double),
            patch("torch.distributed.all_gather_object", side_effect=gather),
            patch("torch.distributed.broadcast_object_list", side_effect=broadcast),
        ):
            self.assertEqual(
                context.reduce_float(
                    5.0,
                    reduction="mean",
                    device=torch.device("cpu"),
                ),
                5.0,
            )
            self.assertEqual(
                context.reduce_float(
                    5.0,
                    reduction="max",
                    device=torch.device("cpu"),
                ),
                10.0,
            )
            self.assertEqual(
                float(context.reduce_tensor_mean(torch.tensor(7.0))),
                7.0,
            )
            self.assertEqual(
                context.all_gather_object("rank-one"),
                ("rank-zero", "rank-one"),
            )
            self.assertEqual(
                context.broadcast_primary_object(None),
                "from-primary",
            )

    def test_session_validates_environment_and_owns_initialized_group(self) -> None:
        with (
            self.assertRaisesRegex(DistributedError, "device"),
            distributed_session("invalid"),
        ):
            self.fail("invalid device unexpectedly yielded a context")

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "bad"}, clear=True),
            self.assertRaisesRegex(DistributedError, "integer"),
            distributed_session("cpu"),
        ):
            self.fail("invalid world size unexpectedly initialized")

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "0"}, clear=True),
            self.assertRaisesRegex(DistributedError, "positive"),
            distributed_session("cpu"),
        ):
            self.fail("invalid world size unexpectedly initialized")

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "2"}, clear=True),
            self.assertRaisesRegex(DistributedError, "incomplete"),
            distributed_session("cpu"),
        ):
            self.fail("incomplete environment unexpectedly initialized")

        environment = {
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "RANK": "1",
            "WORLD_SIZE": "2",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("torch.distributed.is_available", return_value=True),
            patch(
                "torch.distributed.is_initialized",
                side_effect=(False, True),
            ),
            patch("torch.distributed.init_process_group") as initialize,
            patch("torch.distributed.get_rank", return_value=1),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_backend", return_value="gloo"),
            patch("torch.distributed.destroy_process_group") as destroy,
        ):
            with distributed_session("cpu") as context:
                self.assertEqual(context.rank, 1)
                self.assertEqual(context.world_size, 2)
                self.assertTrue(context.initialized_here)
            initialize.assert_called_once_with(backend="gloo", init_method="env://")
            destroy.assert_called_once_with()

    def test_current_reads_an_existing_group(self) -> None:
        with (
            patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=True),
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=3),
            patch("torch.distributed.get_world_size", return_value=4),
            patch("torch.distributed.get_backend", return_value="gloo"),
        ):
            context = DistributedContext.current()
        self.assertEqual(context.rank, 3)
        self.assertEqual(context.local_rank, 3)
        self.assertEqual(context.world_size, 4)


if __name__ == "__main__":
    unittest.main()
