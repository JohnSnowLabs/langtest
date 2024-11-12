import random
from typing import List, Tuple, Union
from langtest.logger import logger
from langtest.transform.robustness import BaseRobustness
from langtest.utils.custom_types.sample import Sample
from PIL import Image, ImageFilter


class ImageResizing(BaseRobustness):
    alias_name = "image_resize"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        resize: Union[float, Tuple[int, int]] = 0.5,
        *args,
        **kwargs,
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_resize"
            if isinstance(resize, float):
                sample.perturbed_image = sample.original_image.resize(
                    (
                        int(sample.original_image.width * resize),
                        int(sample.original_image.height * resize),
                    )
                )
            else:
                sample.perturbed_image = sample.original_image.resize(resize)

        return sample_list


class ImageRotation(BaseRobustness):
    alias_name = "image_rotate"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample], angle: int = 90, exapand=True, *args, **kwargs
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_rotate"
            sample.perturbed_image = sample.original_image.rotate(angle, expand=True)

        return sample_list


class ImageBlur(BaseRobustness):
    alias_name = "image_blur"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample], radius: int = 2, *args, **kwargs
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_blur"
            sample.perturbed_image = sample.original_image.filter(
                ImageFilter.GaussianBlur(radius)
            )

        return sample_list


class ImageNoise(BaseRobustness):
    alias_name = "image_noise"
    supported_tasks = ["visualqa"]

    @classmethod
    def transform(
        cls, sample_list: List[Sample], noise: float = 0.1, *args, **kwargs  # Noise level
    ) -> List[Sample]:
        try:
            if noise < 0 or noise > 1:
                raise ValueError("Noise level must be in the range [0, 1].")

            # Get image size
            for sample in sample_list:
                sample.category = "robustness"
                sample.test_type = "image_noise"
                sample.perturbed_image = cls.add_noise(
                    image=sample.original_image, noise_level=noise
                )
            return sample_list

        except Exception as e:
            logger.error(f"Error in adding noise to the image: {e}")
            raise e

    @staticmethod
    def add_noise(image: Image.Image, noise_level: float) -> Image:
        width, height = image.size

        # Create a new image to hold the noisy version
        noisy_image = image.copy()
        pixels = noisy_image.load()  # Access pixel data

        # Check if the image is grayscale or RGB
        if image.mode == "L":  # Grayscale image
            for x in range(width):
                for y in range(height):
                    # Get the pixel value
                    gray = image.getpixel((x, y))

                    # Generate random noise
                    noise_gray = int(random.gauss(0, 255 * noise_level))

                    # Add noise and clip the value to stay in [0, 255]
                    new_gray = max(0, min(255, gray + noise_gray))

                    # Set the new pixel value
                    pixels[x, y] = new_gray

        elif image.mode == "RGB":  # Color image
            for x in range(width):
                for y in range(height):
                    r, g, b = image.getpixel((x, y))  # Get the RGB values of the pixel

                    # Generate random noise for each channel
                    noise_r = int(random.gauss(0, 255 * noise_level))
                    noise_g = int(random.gauss(0, 255 * noise_level))
                    noise_b = int(random.gauss(0, 255 * noise_level))

                    # Add noise to each channel and clip values to stay in range [0, 255]
                    new_r = max(0, min(255, r + noise_r))
                    new_g = max(0, min(255, g + noise_g))
                    new_b = max(0, min(255, b + noise_b))

                    # Set the new pixel value
                    pixels[x, y] = (new_r, new_g, new_b)

        else:
            raise ValueError("The input image must be in 'L' (grayscale) or 'RGB' mode.")

        return noisy_image


class ImageConstrast(BaseRobustness):
    alias_name = "image_contrast"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample], contrast_factor: float = 0.5, *args, **kwargs
    ) -> List[Sample]:
        from PIL import ImageEnhance

        if contrast_factor < 0:
            raise ValueError("Contrast factor must be above 0.")

        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_contrast"
            img = ImageEnhance.Contrast(sample.original_image)
            sample.perturbed_image = img.enhance(contrast_factor)

        return sample_list


class ImageBrightness(BaseRobustness):
    alias_name = "image_brightness"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample], brightness_factor: float = 0.3, *args, **kwargs
    ) -> List[Sample]:
        from PIL import ImageEnhance

        if brightness_factor < 0:
            raise ValueError("Brightness factor must be above 0.")

        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_brightness"
            enchancer = ImageEnhance.Brightness(sample.original_image)
            sample.perturbed_image = enchancer.enhance(brightness_factor)

        return sample_list


class ImageSharpness(BaseRobustness):
    alias_name = "image_sharpness"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample], sharpness_factor: float = 1.5, *args, **kwargs
    ) -> List[Sample]:
        from PIL import ImageEnhance

        if sharpness_factor < 0:
            raise ValueError("Sharpness factor must be above 0.")

        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_sharpness"
            enchancer = ImageEnhance.Sharpness(sample.original_image)
            sample.perturbed_image = enchancer.enhance(sharpness_factor)

        return sample_list


class ImageColor(BaseRobustness):
    alias_name = "image_color"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample], color_factor: float = 0, *args, **kwargs
    ) -> List[Sample]:
        from PIL import ImageEnhance

        if color_factor < 0:
            raise ValueError("Color factor must be in the range [0, inf].")

        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_color"
            enchancer = ImageEnhance.Color(sample.original_image)
            sample.perturbed_image = enchancer.enhance(color_factor)

        return sample_list


class ImageFlip(BaseRobustness):
    alias_name = "image_flip"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample], flip: str = "horizontal", *args, **kwargs
    ) -> List[Sample]:
        if flip not in ["horizontal", "vertical"]:
            raise ValueError("Flip must be either 'horizontal' or 'vertical'.")

        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_flip"
            if flip == "horizontal":
                sample.perturbed_image = sample.original_image.transpose(
                    Image.FLIP_LEFT_RIGHT
                )
            else:
                sample.perturbed_image = sample.original_image.transpose(
                    Image.FLIP_TOP_BOTTOM
                )

        return sample_list


class ImageCrop(BaseRobustness):
    alias_name = "image_crop"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        crop_size: Union[float, Tuple[int, int]] = (100, 100),
        *args,
        **kwargs,
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_crop"
            if isinstance(crop_size, float):
                sample.perturbed_image = sample.original_image.crop(
                    (
                        0,
                        0,
                        int(sample.original_image.width * crop_size),
                        int(sample.original_image.height * crop_size),
                    )
                )
            else:
                sample.perturbed_image = sample.original_image.crop(
                    (0, 0, crop_size[0], crop_size[1])
                )

        return sample_list
