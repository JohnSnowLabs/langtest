import random
from typing import List, Literal, Tuple, Union
from langtest.logger import logger
from langtest.transform.robustness import BaseRobustness
from langtest.transform.utils import get_default_font
from langtest.utils.custom_types.sample import Sample
from PIL import Image, ImageFilter, ImageDraw


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


class ImageTranslate(BaseRobustness):
    alias_name = "image_translate"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        translate: Tuple[int, int] = (10, 10),
        *args,
        **kwargs,
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_translate"
            sample.perturbed_image = sample.original_image.transform(
                sample.original_image.size,
                Image.AFFINE,
                (1, 0, translate[0], 0, 1, translate[1]),
            )

        return sample_list


class ImageShear(BaseRobustness):
    alias_name = "image_shear"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        shear: float = 0.5,
        *args,
        **kwargs,
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_shear"
            sample.perturbed_image = sample.original_image.transform(
                sample.original_image.size,
                Image.AFFINE,
                (1, shear, 0, 0, 1, 0),
            )

        return sample_list


class ImageBlackSpot(BaseRobustness):
    """
    This class is used to corrupt the image by adding a black box to it.
    """

    alias_name = "image_black_spots"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        max_count: int = 10,
        shape: str = "random",
        size_range: Tuple[int, int] = (10, 50),
        *args,
        **kwargs,
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_black_spots"
            sample.perturbed_image = sample.original_image.copy()
            for _ in range(max_count):
                random_size = random.randint(*size_range)
                # get random values for the black box
                x1 = random.randint(0, sample.original_image.width - random_size)
                y1 = random.randint(0, sample.original_image.height - random_size)
                x2 = x1 + random_size
                y2 = y1 + random_size

                opacity = random.uniform(0.5, 1)

                if shape == "random":
                    shapes = ["rectangle", "circle"]
                    random.shuffle(shapes)
                    selected_shape = random.choice(shapes)
                else:
                    selected_shape = shape

                if selected_shape == "rectangle":
                    mask = Image.new("RGBA", sample.original_image.size, (0, 0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask)
                    mask_draw.rectangle(
                        (x1, y1, x2, y2), fill=(0, 0, 0, int(255 * opacity))
                    )
                    sample.perturbed_image.paste(mask, (0, 0), mask)
                elif selected_shape == "circle":
                    mask = Image.new("RGBA", sample.original_image.size, (0, 0, 0, 0))
                    mask_draw = ImageDraw.Draw(mask)
                    mask_draw.ellipse(
                        (x1, y1, x2, y2), fill=(0, 0, 0, int(255 * opacity))
                    )
                    sample.perturbed_image.paste(mask, (0, 0), mask)

                else:
                    raise ValueError("Shape must be either 'rectangle' or 'circle'.")

        return sample_list


class ImageLayeredMask(BaseRobustness):
    alias_name = "image_layered_mask"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        mask: Union[Image.Image, str, None] = None,
        opacity: float = 0.5,
        flip: Literal["horizontal", "vertical"] = "horizontal",
        *args,
        **kwargs,
    ) -> List[Sample]:
        reset_mask = mask
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_layered_mask"
            sample.perturbed_image = sample.original_image.copy()
            if mask is None:
                mask = sample.original_image

            elif isinstance(mask, str):
                mask = Image.open(mask)

            if flip == "horizontal":
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip == "vertical":
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            mask = mask.convert("RGBA")
            mask.putalpha(int(255 * opacity))
            sample.perturbed_image.paste(mask, (0, 0), mask)

            mask = reset_mask

        return sample_list


class ImageTextOverlay(BaseRobustness):
    alias_name = "image_text_overlay"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        text: str = "LangTest",
        font_size: int = 100,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        *args,
        **kwargs,
    ) -> List[Sample]:
        from PIL import ImageFont

        # transperant text overlay on the image
        font_color = font_color + (255,)

        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_text_overlay"
            sample.perturbed_image = sample.original_image.copy()
            draw = ImageDraw.Draw(sample.perturbed_image)
            font = get_default_font(font_size)

            draw.text(
                (sample.original_image.width // 2, sample.original_image.height // 2),
                text,
                font=font,
                fill=font_color,
            )

        return sample_list


class ImageWatermark(BaseRobustness):
    alias_name = "image_watermark"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        watermark: Union[Image.Image, str] = None,
        position: Tuple[int, int] = (10, 10),
        opacity: float = 0.5,
        *args,
        **kwargs,
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_watermark"
            sample.perturbed_image = sample.original_image.copy()

            if watermark is None:
                # If no watermark is provided, add a random text as watermark
                watermark = Image.new("RGBA", sample.original_image.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(watermark)
                draw.text(
                    position,
                    "LangTest",
                    font=None,
                    fill=(255, 255, 255, int(255 * opacity)),
                )
            elif isinstance(watermark, str):
                watermark = Image.open(watermark)
                watermark = watermark.convert("RGBA")
                watermark.putalpha(int(255 * opacity))
            else:
                watermark = watermark.convert("RGBA")
                watermark.putalpha(int(255 * opacity))

            sample.perturbed_image.paste(watermark, (0, 0), watermark)

        return sample_list


class ImageRandomTextOverlay(BaseRobustness):
    alias_name = "image_random_text_overlay"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        opacity: float = 0.5,
        font_size: int = 30,
        random_texts: int = 10,
        color: Tuple[int, int, int] = (0, 0, 0),
        *args,
        **kwargs,
    ) -> List[Sample]:
        from PIL import ImageFont

        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_random_text_overlay"
            sample.perturbed_image = sample.original_image.copy()
            overlay = Image.new("RGBA", sample.original_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for _ in range(random_texts):
                font = get_default_font(font_size)
                x1 = random.randint(0, sample.original_image.width)
                y1 = random.randint(0, sample.original_image.height)

                draw.text(
                    (x1, y1),
                    "LangTest",
                    font=font,
                    fill=(*color, int(255 * opacity)),
                )
            sample.perturbed_image.paste(overlay, (0, 0), overlay)

        return sample_list


class ImageRandomLineOverlay(BaseRobustness):
    alias_name = "image_random_line_overlay"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        color: Tuple[int, int, int] = (255, 0, 0),
        opacity: float = 0.5,
        random_lines: int = 10,
        *args,
        **kwargs,
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_random_line_overlay"
            sample.perturbed_image = sample.original_image.copy()
            overlay = Image.new("RGBA", sample.original_image.size)
            overlay.putalpha(int(255 * opacity))
            draw = ImageDraw.Draw(overlay)

            for _ in range(random_lines):
                # Get random points for the line
                x1 = random.randint(0, sample.original_image.width)
                y1 = random.randint(0, sample.original_image.height)
                x2 = random.randint(0, sample.original_image.width)
                y2 = random.randint(0, sample.original_image.height)

                draw.line(
                    [(x1, y1), (x2, y2)],
                    fill=color + (int(255 * opacity),),
                    width=5,
                )
            sample.perturbed_image.paste(overlay, (0, 0), overlay)

        return sample_list


class ImageRandomPolygonOverlay(BaseRobustness):
    alias_name = "image_random_polygon_overlay"
    supported_tasks = ["visualqa"]

    @staticmethod
    def transform(
        sample_list: List[Sample],
        color: Tuple[int, int, int] = (255, 0, 0),
        opacity: float = 0.2,
        random_polygons: int = 10,
        *args,
        **kwargs,
    ) -> List[Sample]:
        for sample in sample_list:
            sample.category = "robustness"
            sample.test_type = "image_random_polygon_overlay"
            sample.perturbed_image = sample.original_image.copy()
            overlay = Image.new("RGBA", sample.original_image.size)
            overlay.putalpha(int(255 * opacity))
            draw = ImageDraw.Draw(overlay)

            for _ in range(random_polygons):
                # Get random points for the polygon vertices with random vertices
                vertices = [
                    (
                        random.randint(0, sample.original_image.width),
                        random.randint(0, sample.original_image.height),
                    )
                    for _ in range(random.randint(3, 6))
                ]

                draw.polygon(
                    vertices,
                    fill=color + (int(255 * opacity),),
                )
            sample.perturbed_image.paste(overlay, (0, 0), overlay)

        return sample_list
