import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "01_data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "01_data" / "processed"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def create_directories() -> None:

    for split in ["train", "validation", "test"]:
        for class_name in ["cats", "dogs"]:
            directory = PROCESSED_DATA_DIR / split / class_name
            directory.mkdir(parents=True, exist_ok=True)


def get_image_files(directory: Path) -> list[Path]:

    return [
        file
        for file in directory.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def split_class_images(class_name: str) -> None:

    source_directory = RAW_DATA_DIR / class_name
    images = get_image_files(source_directory)

    random.shuffle(images)

    total_images = len(images)

    train_end = int(total_images * TRAIN_RATIO)
    validation_end = train_end + int(total_images * VALIDATION_RATIO)

    train_images = images[:train_end]
    validation_images = images[train_end:validation_end]
    test_images = images[validation_end:]

    splits = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images,
    }

    for split_name, split_images in splits.items():
        destination_directory = (
            PROCESSED_DATA_DIR / split_name / class_name
        )

        for image_path in split_images:
            destination_path = destination_directory / image_path.name
            shutil.copy2(image_path, destination_path)

    print(f"\nClass: {class_name}")
    print(f"Total: {total_images}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(validation_images)}")
    print(f"Test: {len(test_images)}")


def main() -> None:
    random.seed(RANDOM_SEED)

    create_directories()

    split_class_images("cats")
    split_class_images("dogs")

    print("\nDataset splitted successfully!")


if __name__ == "__main__":
    main()