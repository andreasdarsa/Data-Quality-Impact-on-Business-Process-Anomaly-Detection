from src.generation.log_gen import generate_log
from src.data.loader import save_logs
from src.data.schema import validate_logs
from src.utils.paths import RAW_DATA_DIR

from experiments.configs import SEEDS, get_raw_log_filename


def generate_dataset_for_seed(seed: int) -> None:
    logs = generate_log(
        n_normal=8000,
        anomaly_counts={
            "structural": 400,
            "temporal": 400,
            "contextual": 400,
        },
        seed=seed,
    )

    validate_logs(logs)

    output_path = RAW_DATA_DIR / get_raw_log_filename(seed)
    save_logs(logs, output_path)

    print(f"[seed={seed}] Dataset created: {len(logs)} cases")
    print(f"[seed={seed}] Saved to: {output_path}")


def main() -> None:
    for seed in SEEDS:
        generate_dataset_for_seed(seed)


if __name__ == "__main__":
    main()
