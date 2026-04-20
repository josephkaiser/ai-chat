import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "src" / "python" / "harness.py"


class TimestampNormalizationTests(unittest.TestCase):
    def test_harness_persists_utc_iso_timestamps(self):
        harness = HARNESS_PATH.read_text(encoding="utf-8")

        self.assertIn("from datetime import datetime, timedelta, timezone", harness)
        self.assertIn("return datetime.now(timezone.utc).isoformat()", harness)
        self.assertIn("return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()", harness)
        self.assertNotIn("datetime.now().isoformat()", harness)

    def test_harness_parses_legacy_and_utc_offset_timestamps(self):
        harness = HARNESS_PATH.read_text(encoding="utf-8")

        self.assertIn('datetime.fromisoformat(str(value).replace("Z", "+00:00"))', harness)
        self.assertIn("if parsed.tzinfo is None:", harness)
        self.assertIn("return parsed.astimezone()", harness)


if __name__ == "__main__":
    unittest.main()
