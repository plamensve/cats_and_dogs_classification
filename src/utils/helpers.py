import numpy as np
from PIL import Image
import pandas as pd


class DatasetImageInfo:
    def __init__(self, dataset: list):
        """
        :param dataset: list with paths of images
        """
        self.dataset = dataset

        self.image_sizes = []
        self.x_sizes = []
        self.y_sizes = []

    def dataset_image_sizes(self):
        self.image_sizes = []

        for image_path in self.dataset:
            with Image.open(image_path) as image:
                self.image_sizes.append(image.size)

        return self.image_sizes

    def dataset_info(self):
        if not self.image_sizes:
            self.dataset_image_sizes()

        self.x_sizes = []
        self.y_sizes = []

        for width, height in self.image_sizes:
            self.x_sizes.append(width)
            self.y_sizes.append(height)

        return self.x_sizes, self.y_sizes

    def dataset_image_info(self):
        if not self.x_sizes or not self.y_sizes:
            self.dataset_info()

        x_size_mean = np.mean(self.x_sizes)
        y_size_mean = np.mean(self.y_sizes)

        x_size_min = np.min(self.x_sizes)
        y_size_min = np.min(self.y_sizes)

        x_size_max = np.max(self.x_sizes)
        y_size_max = np.max(self.y_sizes)

        return pd.DataFrame({
            "X Mean": [x_size_mean],
            "Y Mean": [y_size_mean],
            "X Max": [x_size_max],
            "Y Max": [y_size_max],
            "X Min": [x_size_min],
            "Y Min": [y_size_min]
        })