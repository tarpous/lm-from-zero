from __future__ import annotations

import io
import os
import unittest
from unittest.mock import patch

from lm_from_zero.progress import ProgressReporter, progress_enabled


class ProgressTests(unittest.TestCase):
    def test_default_detection_and_invalid_refresh_interval(self) -> None:
        stream = io.StringIO()
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(progress_enabled(stream))
        with self.assertRaises(ValueError):
            ProgressReporter("invalid", enabled=False, refresh_seconds=0)

    def test_terminal_override_and_rendered_phase(self) -> None:
        stream = io.StringIO()
        with patch.dict(os.environ, {"LM_FROM_ZERO_PROGRESS": "1"}):
            self.assertTrue(progress_enabled(stream))
            reporter = ProgressReporter(
                "training",
                enabled=True,
                stream=stream,
                refresh_seconds=0.01,
            )
            reporter.phase("compile", fields={"mode": "default"})
            reporter.phase("training", total=3, current=1)
            reporter.update(2, fields={"loss": 1.25, "tokens": 16_384})
            reporter.update(2)
            reporter.advance(fields={"ratio": 0.5})
            reporter.finish("complete")
            reporter.finish("ignored")

        output = stream.getvalue()
        self.assertIn("training | compile", output)
        self.assertIn("training | complete", output)
        self.assertIn("2/3", output)
        self.assertIn("loss=1.25", output)

    def test_disabled_progress_is_silent(self) -> None:
        stream = io.StringIO()
        with patch.dict(os.environ, {"LM_FROM_ZERO_PROGRESS": "0"}):
            self.assertFalse(progress_enabled(stream))
            reporter = ProgressReporter("quiet", stream=stream)
            reporter.phase("work", total=2)
            reporter.update(1)
            reporter.finish()
        self.assertEqual(stream.getvalue(), "")

    def test_invalid_progress_movement_is_rejected(self) -> None:
        reporter = ProgressReporter("validation", enabled=False)
        with self.assertRaises(ValueError):
            reporter.phase("work", total=0)
        with self.assertRaises(ValueError):
            reporter.phase("work", total=2, current=-1)
        with self.assertRaises(ValueError):
            reporter.phase("work", total=2, current=3)
        reporter.phase("work", total=2)
        with self.assertRaises(ValueError):
            reporter.update(-1)
        reporter.update(1)
        with self.assertRaises(ValueError):
            reporter.update(0)
        with self.assertRaises(ValueError):
            reporter.advance(-1)

    def test_ticker_stops_when_finish_is_already_requested(self) -> None:
        reporter = ProgressReporter("ticker", enabled=False, refresh_seconds=0.01)
        reporter._finished = True
        reporter._stop.clear()
        reporter._ticker()
        reporter._finished = False
        reporter.finish()


if __name__ == "__main__":
    unittest.main()
