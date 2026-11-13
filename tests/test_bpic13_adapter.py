
import os
import unittest

from src.data.adapters.bpic13_adapter import (
    convert_bpic13_case,
    convert_bpic13_log,
    load_and_convert_bpic13_log,
)
from src.data.schema import validate_logs
from src.data.adapters.binet_adapter import (
    load_and_convert_binet_log
)


def make_case(case_id="case-1", label="normal", events=None):
    return {
        "id": case_id,
        "attributes": {"label": label},
        "events": events if events is not None else [
            {
                "name": "Activity A",
                "timestamp": "2013-01-01T10:00:00+01:00",
                "attributes": {
                    "concept:name": "Activity A",
                    "org:resource": "resource-1",
                    "lifecycle:transition": "Closed",
                },
            }
        ],
    }


class TestBPIC13Adapter(unittest.TestCase):

    def test_normal_label(self):
        converted = convert_bpic13_case(make_case())

        self.assertFalse(converted["is_anomaly"])
        self.assertEqual(converted["subtype"], "normal")

    def test_anomaly_label(self):
        label = {
            "anomaly": "Insert",
            "attr": {"indices": [0]}
        }

        converted = convert_bpic13_case(
            make_case(label=label)
        )

        self.assertTrue(converted["is_anomaly"])
        self.assertEqual(converted["subtype"], "Insert")

    def test_missing_timestamp(self):
        event = {
            "name": "Activity A",
            "timestamp": None,
            "attributes": {}
        }

        converted = convert_bpic13_case(
            make_case(events=[event])
        )

        result = converted["events"][0]

        self.assertIsNone(result["timestamp_start"])
        self.assertIsNone(result["timestamp_end"])
        self.assertIsNone(result["duration"])

    def test_activity_fallback(self):
        event = {
            "name": "Fallback activity",
            "timestamp": None,
            "attributes": {}
        }

        converted = convert_bpic13_case(
            make_case(events=[event])
        )

        self.assertEqual(
            converted["events"][0]["activity"],
            "Fallback activity"
        )

    def test_preserves_event_order(self):
        events = [
            {
                "name": "B",
                "timestamp": "2020-01-02T00:00:00+00:00",
                "attributes": {"concept:name": "B"}
            },
            {
                "name": "A",
                "timestamp": "2020-01-01T00:00:00+00:00",
                "attributes": {"concept:name": "A"}
            }
        ]

        converted = convert_bpic13_case(
            make_case(events=events)
        )

        activities = [
            event["activity"]
            for event in converted["events"]
        ]

        self.assertEqual(activities, ["B", "A"])

    def test_timestamp_fallback(self):
        event = {
            "name": "Activity A",
            "attributes": {
                "concept:name": "Activity A",
                "time:timestamp": "2020-01-01T00:00:00+00:00"
            }
        }

        converted = convert_bpic13_case(
            make_case(events=[event])
        )

        self.assertEqual(
            converted["events"][0]["timestamp_end"],
            "2020-01-01T00:00:00+00:00"
        )

    def test_total_duration(self):
        events = [
            {
                "name": "A",
                "timestamp": "2020-01-01T00:00:00+00:00",
                "attributes": {}
            },
            {
                "name": "B",
                "timestamp": "2020-01-01T01:00:00+00:00",
                "attributes": {}
            }
        ]

        converted = convert_bpic13_case(
            make_case(events=events)
        )

        self.assertEqual(
            converted["total_duration"],
            3600.0
        )

    def test_schema_compatibility(self):
        converted = convert_bpic13_case(make_case())

        validate_logs([converted])

    def test_duplicate_case_ids(self):
        case = make_case(case_id="duplicate")

        with self.assertRaises(ValueError):
            convert_bpic13_log({
                "cases": [case, case]
            })

    def test_empty_case(self):
        with self.assertRaises(ValueError):
            convert_bpic13_case(
                make_case(events=[])
            )

    def test_real_dataset(self):
        logs = load_and_convert_bpic13_log(
            "data/raw/bpic13-0.3-1.json"
        )

        validate_logs(logs)

        self.assertEqual(len(logs), 1487)

        self.assertEqual(
            sum(len(case["events"]) for case in logs),
            6891
        )

        self.assertEqual(
            sum(case["is_anomaly"] for case in logs),
            362
        )

        self.assertTrue(
            all(
                event["duration"] is None
                for case in logs
                for event in case["events"]
            )
        )
        
    def test_bpic12_regression(self):
        from src.data.adapters.binet_adapter import (
            load_and_convert_binet_log
        )

        logs = load_and_convert_binet_log(
            "data/raw/bpic12-0.3-1.json"
        )

        validate_logs(logs)

        self.assertEqual(len(logs), 13087)


if __name__ == "__main__":
    unittest.main()
