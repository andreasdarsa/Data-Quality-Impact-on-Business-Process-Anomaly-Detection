
import unittest

from src.features.temporal import extract_temporal_features


class TestTemporalFeatures(unittest.TestCase):

    def test_bpic12_gap(self):
        case = {
            "events": [
                {
                    "timestamp_start": "2020-01-01T10:00:00",
                    "timestamp_end": "2020-01-01T10:30:00",
                    "duration": 1800
                },
                {
                    "timestamp_start": "2020-01-01T11:00:00",
                    "timestamp_end": "2020-01-01T11:30:00",
                    "duration": 1800
                }
            ]
        }

        features = extract_temporal_features(case)

        self.assertEqual(features["mean_gap"], 30.0)

    def test_bpic13_gap(self):
        case = {
            "events": [
                {
                    "timestamp_start": None,
                    "timestamp_end": "2020-01-01T10:00:00",
                    "duration": None
                },
                {
                    "timestamp_start": None,
                    "timestamp_end": "2020-01-01T10:20:00",
                    "duration": None
                }
            ]
        }

        features = extract_temporal_features(case)

        self.assertEqual(features["mean_gap"], 20.0)

    def test_negative_gap_preserved(self):
        case = {
            "events": [
                {
                    "timestamp_start": None,
                    "timestamp_end": "2020-01-01T10:30:00",
                    "duration": None
                },
                {
                    "timestamp_start": None,
                    "timestamp_end": "2020-01-01T10:00:00",
                    "duration": None
                }
            ]
        }

        features = extract_temporal_features(case)

        self.assertEqual(features["mean_gap"], -30.0)
        self.assertEqual(features["min_gap"], -30.0)

    def test_missing_timestamps(self):
        case = {
            "events": [
                {
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "duration": None
                },
                {
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "duration": None
                }
            ]
        }

        features = extract_temporal_features(case)

        self.assertEqual(features["mean_gap"], 0.0)
        self.assertEqual(features["mean_duration"], 0.0)

    def test_missing_durations(self):
        case = {
            "events": [
                {
                    "timestamp_start": None,
                    "timestamp_end": "2020-01-01T10:00:00",
                    "duration": None
                },
                {
                    "timestamp_start": None,
                    "timestamp_end": "2020-01-01T10:10:00",
                    "duration": None
                }
            ]
        }

        features = extract_temporal_features(case)

        self.assertEqual(features["mean_duration"], 0.0)
        self.assertEqual(features["max_duration"], 0.0)


if __name__ == "__main__":
    unittest.main()
