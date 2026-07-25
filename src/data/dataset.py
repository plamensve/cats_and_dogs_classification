from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "01_data" / "processed"

IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 0


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Create image transformations for training and evaluation."""

    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    evaluation_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    return train_transforms, evaluation_transforms


def create_datasets():
    """Create training, validation, and test datasets."""

    train_transforms, evaluation_transforms = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "train",
        transform=train_transforms,
    )

    validation_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "validation",
        transform=evaluation_transforms,
    )

    test_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "test",
        transform=evaluation_transforms,
    )

    return train_dataset, validation_dataset, test_dataset


def create_data_loaders():
    """Create data loaders for training, validation, and testing."""

    train_dataset, validation_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, validation_loader, test_loader


def main() -> None:
    train_loader, validation_loader, test_loader = create_data_loaders()

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Class mapping: {train_loader.dataset.class_to_idx}")


if __name__ == "__main__":
    main()