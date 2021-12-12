import imutils
from cv2 import imread, imwrite
from matplotlib import pyplot as plt

import skimage as sk
from skimage import transform
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian

from numpy import ndarray
import random
import os
import numpy as np
import cv2

class AugmentationGenerator:

    @staticmethod
    def start(image_to_transform: ndarray, save_folder_path: str, image_amount: int = 25):

        image_to_transform = image_to_transform / 255.0
        original_image = np.copy(image_to_transform)
        #image_to_transform = imutils.resize(image_to_transform, width=600, inter=cv2.INTER_CUBIC)
        file_path = os.path.join(save_folder_path, "original_image.jpg")
        AugmentationGenerator.save_image(original_image * 255.0, file_path)

        for num_generated_file in range(image_amount):

            #image_to_transform = np.copy(original_image)
            # random num of transformations to apply
            available_transformations = {
                'rotate': AugmentationGenerator.add_random_rotation,
                'horizontal_flip': AugmentationGenerator.add_horizontal_flip,
                'noise': AugmentationGenerator.add_random_noise,
                'shear': AugmentationGenerator.add_shear,
                'horizontal_shift': AugmentationGenerator.add_horizontal_shift,
                'vertical_shift': AugmentationGenerator.add_vertical_shift,
                'zoom': AugmentationGenerator.add_scale,
                'blur': AugmentationGenerator.add_blur
            }
            num_transformations_to_apply = random.randint(1, len(available_transformations))

            num_transformations = 0
            transformed_image = np.copy(original_image)

            while available_transformations and num_transformations <= num_transformations_to_apply:
                # choose a random transformation to apply for a single image

                key = random.choice(list(available_transformations))
                transformed_image = available_transformations[key](transformed_image)
                num_transformations += 1

                del available_transformations[key]

            file_path = os.path.join(save_folder_path, f"augmented_{num_generated_file + 1}.jpg")
            AugmentationGenerator.save_image(transformed_image * 255.0, file_path)


    @staticmethod
    def save_image(transformed_image: ndarray, file_path: str):

        if transformed_image is not None:
            print(f"[INFO] Generated augmented image: {file_path}")
            imwrite(file_path, transformed_image)
        else:
            print('image is not defined!')

    @staticmethod
    def add_shear(image_array: ndarray):
        # image shearing using sklearn.transform.AffineTransform
        # try out with differnt values of shear
        shear_level = random.uniform(-15, 15) / 100
        tf = AffineTransform(shear=shear_level)
        return transform.warp(image_array, tf, order=1, preserve_range=True, mode='wrap')

    @staticmethod
    def add_scale(image_array: ndarray):
        # Image rescaling with sklearn.transform.rescale
        scale = 1.0 + random.uniform(-15, 15) / 100

        return transform.rescale(image_array, scale)

    @staticmethod
    def _add_shift(image_array: ndarray, vector):
        transform = AffineTransform(translation=vector)
        shifted = warp(image_array, transform, mode='wrap', preserve_range=True)

        return shifted.astype(image_array.dtype)

    @staticmethod
    def add_horizontal_shift(image_array: ndarray, offset: float=0.2):
        vector = (int(image_array.shape[1] / offset), 0)
        return AugmentationGenerator._add_shift(image_array, vector)

    @staticmethod
    def add_vertical_shift(image_array: ndarray, offset: float=0.2):
        vector = (0, int(image_array.shape[0] / offset))
        return AugmentationGenerator._add_shift(image_array, vector)


    @staticmethod
    def add_random_rotation(image_array: ndarray):
        # pick a random degree of rotation between 25% on the left and 25% on the right
        random_degree = random.uniform(-25, 25)
        return sk.transform.rotate(image_array, random_degree)

    @staticmethod
    def add_random_noise(image_array: ndarray):
        # add random noise to the image

        return sk.util.random_noise(image_array)

    @staticmethod
    def add_horizontal_flip(image_array: ndarray):
        return image_array[:, ::-1]

    @staticmethod
    def add_blur(image_array: ndarray, is_random: bool=True):
        sigma = 1.0
        random_seed = 0.0
        if is_random:
            random_seed = random.random()

        return sk.filters.gaussian(
            image_array, sigma=sigma, truncate=3.5, multichannel=True)

    @staticmethod
    def show_image(image_array: ndarray, image_title: str='Image'):
        plt.imshow(image_array)
        plt.title(image_title)
        plt.show()

