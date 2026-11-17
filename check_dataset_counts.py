from collections import Counter

from src.data.adapters.binet_adapter import load_and_convert_binet_log
from src.data.adapters.bpic13_adapter import load_and_convert_bpic13_log


def report(name, logs):
    total = len(logs)
    anomalous = sum(c["is_anomaly"] for c in logs)
    normal = total - anomalous

    subtypes = Counter(
        c["subtype"]
        for c in logs
        if c["is_anomaly"]
    )

    events = sum(len(c["events"]) for c in logs)

    print(f"\n{'=' * 45}")
    print(name)
    print(f"{'=' * 45}")
    print(f"Cases:     {total:,}")
    print(f"Events:    {events:,}")
    print(f"Normal:    {normal:,}")
    print(f"Anomalous: {anomalous:,}")
    print(f"Anomaly ratio: {anomalous / total:.2%}")

    print("\nAnomaly subtypes:")
    for subtype, count in sorted(subtypes.items()):
        print(f"{subtype:<30} {count:>6,}")


bpic12 = load_and_convert_binet_log(
    "data/raw/bpic12-0.3-1.json"
)

bpic13 = load_and_convert_bpic13_log(
    "data/raw/bpic13-0.3-1.json"
)

report("BPIC12", bpic12)
report("BPIC13", bpic13)
