from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Get the absolute path to the main project directory.
#
# __file__ points to the current Python file, for example:
# cats_and_dogs_classification/src/data/dataset.py
#
# .resolve() converts it to an absolute path.
#
# .parents[2] moves two directory levels upward:
# dataset.py -> data -> src -> cats_and_dogs_classification
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Define the directory containing the processed dataset.
PROCESSED_DATA_DIR = PROJECT_ROOT / "01_data" / "processed"


# Define the width and height to which every image will be resized.
#
# Neural networks process images in batches, so all images in one batch
# must have the same dimensions.
IMAGE_SIZE = 128

# Define how many images will be processed together in one batch.
#
# A batch size of 32 means that the model will receive 32 images
# during each training iteration.
BATCH_SIZE = 32

# Define how many additional worker processes will load the data.
#
# NUM_WORKERS = 0 means that data loading will happen in the main process.
# This setting is usually safer on Windows and avoids multiprocessing issues.
NUM_WORKERS = 0

def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Create image transformations for training and evaluation.

    Training transformations include data augmentation because the model
    should see slightly different versions of the same training images.

    Evaluation transformations do not include random augmentation because
    validation and test results must remain consistent and reproducible.

    Returns:
        tuple[transforms.Compose, transforms.Compose]:
            The training transformations and evaluation transformations.
    """

    train_transforms = transforms.Compose(
        [
            # Convert every image to RGB format.
            #
            # Some images in the dataset may use grayscale, palette,
            # RGBA, or CMYK modes. Converting all images to RGB ensures
            # that every image has exactly three color channels.
            transforms.Lambda(lambda image: image.convert("RGB")),

            # Resize every image to 128 x 128 pixels.
            #
            # The original image files are not modified.
            # Resizing happens only when an image is loaded.
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

            # Randomly flip some training images horizontally.
            #
            # This is a data augmentation technique. It creates additional
            # visual variation and helps the model generalize better.
            transforms.RandomHorizontalFlip(),

            # Randomly rotate training images by up to 10 degrees.
            #
            # This allows the model to learn that small changes in image
            # orientation should not change whether the image contains
            # a cat or a dog.
            transforms.RandomRotation(10),

            # Convert the PIL image into a PyTorch tensor.
            #
            # The tensor shape becomes:
            # [channels, height, width]
            #
            # In this project:
            # [3, 128, 128]
            #
            # Pixel values are also converted from the range [0, 255]
            # to the range [0.0, 1.0].
            transforms.ToTensor(),

            # Normalize the image tensor.
            #
            # The normalization formula is:
            #
            # normalized_value = (pixel_value - mean) / std
            #
            # Because the tensor values are initially between 0 and 1,
            # using mean=0.5 and std=0.5 transforms them approximately
            # into the range [-1, 1].
            #
            # One mean and standard deviation value is provided
            # for each RGB channel.
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
        ]
    )

    evaluation_transforms = transforms.Compose(
        [
            # Convert every validation and test image to RGB format.
            transforms.Lambda(lambda image: image.convert("RGB")),

            # Resize every validation and test image to 128 x 128 pixels.
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

            # Convert the image to a PyTorch tensor.
            transforms.ToTensor(),

            # Apply the same normalization used for training images.
            #
            # Training, validation, and test images must use the same
            # normalization so that the model receives data on the same scale.
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],),
        ]
    )

    return train_transforms, evaluation_transforms


def create_datasets() -> tuple[
    datasets.ImageFolder,
    datasets.ImageFolder,
    datasets.ImageFolder,
]:
    """
    Create training, validation, and test datasets.
    ImageFolder automatically assigns labels based on subdirectory names.
    Returns:
        tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
            The training, validation, and test datasets.
    """

    # Get the transformations for training and evaluation.
    train_transforms, evaluation_transforms = get_transforms()

    # Create the training dataset.
    #
    # ImageFolder reads images from the class subdirectories and applies
    # the training transformations every time an image is loaded.
    train_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "train",
        transform=train_transforms,
    )

    # Create the validation dataset.
    #
    # Random augmentation is not used here because validation metrics
    # must be calculated on stable and consistent images.
    validation_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "validation",
        transform=evaluation_transforms,
    )

    # Create the test dataset.
    #
    # The test dataset uses the same deterministic transformations
    # as the validation dataset.
    test_dataset = datasets.ImageFolder(
        root=PROCESSED_DATA_DIR / "test",
        transform=evaluation_transforms,
    )

    return train_dataset, validation_dataset, test_dataset


def create_data_loaders() -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Create data loaders for training, validation, and testing.

    A DataLoader:
    - divides the dataset into batches;
    - optionally shuffles the data;
    - loads images when they are needed;
    - returns image tensors together with their labels.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
            The training, validation, and test data loaders.
    """

    # Create the three datasets.
    train_dataset, validation_dataset, test_dataset = create_datasets()

    # Create the training DataLoader.
    train_loader = DataLoader(
        dataset=train_dataset,

        # Load 32 images at a time.
        batch_size=BATCH_SIZE,

        # Shuffle the training images before each epoch.
        #
        # This prevents the model from learning patterns caused by
        # the fixed order of the images.
        shuffle=True,

        # Load data in the main process.
        num_workers=NUM_WORKERS,
    )

    # Create the validation DataLoader.
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,

        # Validation data does not need to be shuffled.
        #
        # The model does not learn from these images, so their order
        # does not affect the validation results.
        shuffle=False,

        num_workers=NUM_WORKERS,
    )

    # Create the test DataLoader.
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,

        # Test data also remains in a fixed order.
        shuffle=False,

        num_workers=NUM_WORKERS,
    )

    return train_loader, validation_loader, test_loader


def main() -> None:
    """
    Create the data loaders and verify that they work correctly.
    This function is used as a simple test before training the model.
    """

    # Create the training, validation, and test DataLoaders.
    train_loader, validation_loader, test_loader = create_data_loaders()

    # Print the number of batches in each DataLoader.
    #
    # The number of batches is approximately:
    #
    # number_of_images / batch_size
    #
    # The final batch may contain fewer than 32 images.
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create an iterator from the training DataLoader and retrieve
    # the first batch of images and labels.
    images, labels = next(iter(train_loader))

    # Print the shape of the image batch.
    #
    # Expected shape:
    # [batch_size, channels, height, width]
    #
    # In this project:
    # [32, 3, 128, 128]
    print(f"Image batch shape: {images.shape}")

    # Print the shape of the label tensor.
    #
    # Expected shape:
    # [batch_size]
    #
    # Each image has one corresponding class label.
    print(f"Label batch shape: {labels.shape}")

    # Print the class-name-to-label mapping created by ImageFolder.
    #
    # Expected result:
    # {'cats': 0, 'dogs': 1}
    print(f"Class mapping: {train_loader.dataset.class_to_idx}")


# Run main() only when this file is executed directly.
#
# The function will not run automatically when this file is imported
# into another module, such as the future training script.
if __name__ == "__main__":
    main()