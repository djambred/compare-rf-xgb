import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_ID = "manueltonneau/indonesian-hate-speech-superset"
OUTPUT_DIR = Path("dataset")
FILE_PREFIX = "indonesian_hate_speech_superset_"


def dataset_already_exists() -> bool:
    return any(OUTPUT_DIR.glob(f"{FILE_PREFIX}*.csv"))


def _copy_supported_files(snapshot_dir: Path) -> int:
    supported_patterns = ["*.csv", "*.parquet", "*.json", "*.jsonl"]
    files_to_copy: list[Path] = []

    for pattern in supported_patterns:
        files_to_copy.extend(snapshot_dir.rglob(pattern))

    copied_count = 0
    for source_path in files_to_copy:
        target_name = source_path.name
        if not target_name.startswith(FILE_PREFIX):
            target_name = f"{FILE_PREFIX}{target_name}"

        target_path = OUTPUT_DIR / target_name
        shutil.copy2(source_path, target_path)
        copied_count += 1
        print(f"Saved file to {target_path}")

    return copied_count


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    force_download = os.getenv("FORCE_DOWNLOAD", "false").lower() == "true"
    if dataset_already_exists() and not force_download:
        print("Dataset already exists. Skipping download.")
        print("Set FORCE_DOWNLOAD=true to re-download.")
        return

    print(f"Downloading dataset snapshot: {DATASET_ID}")
    hf_token = os.getenv("HF_TOKEN") or None
    snapshot_path = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=hf_token,
        allow_patterns=["*.csv", "*.parquet", "*.json", "*.jsonl", "README*"],
    )

    copied_count = _copy_supported_files(Path(snapshot_path))
    if copied_count == 0:
        raise RuntimeError("No supported dataset files found in snapshot.")

    print(f"Completed. Total files copied: {copied_count}")


if __name__ == "__main__":
    main()
