from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, UnidentifiedImageError

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# DVC cache check: comment added
def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()` or equivalent);
      5. no split or class is empty.

    TODO: implement the checks and return a JSON-serializable report.
    The report must not contain timestamps because it is a DVC output.
    """
    report = {"splits": {}}
    total = 0

    for split, expected_count in EXPECTED_SPLITS.items():
        split_dir = raw_root / split

        # Check if split folder exists
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split: {split}")
        
        # check if class folders are exactly the expected three
        found_classes = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if found_classes != EXPECTED_CLASSES:
            raise RuntimeError(f"{split}: classes {found_classes}, expected {EXPECTED_CLASSES}")

        class_counts = {}
        for cls in sorted(EXPECTED_CLASSES):
            class_dir = split_dir / cls 
            images = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            # check images
            for path in images:
                try:
                    with Image.open(path) as img:
                        img.load()
                except Exception as exc:
                    raise UnidentifiedImageError(f"Bad image: {path}") from exc

            if len(images) == 0:
                raise RuntimeError(f"{split}/{cls} is empty")
            class_counts[cls] = len(images)

        # Check 3
        split_total = sum(class_counts.values())                       
        if split_total != expected_count:                       
            raise RuntimeError(f"{split}: {split_total} images, expected {expected_count}")

        report["splits"][split] = {"images": split_total, "class_counts": class_counts}
        total += split_total

    report["status"] = "PASS"
    report["classes"] = sorted(EXPECTED_CLASSES)                
    report["total_images"] = total
    return report


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
