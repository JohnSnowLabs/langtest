import pytest 
from PIL import Image
import requests

from langtest.transform.image import robustness as image_robustness
from langtest.transform.robustness import BaseRobustness

from langtest.utils.custom_types.sample import VisualQASample

class TestImageRobustness:

    url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    @pytest.mark.parametrize(
        "robustness",
        [
           test_type for name, test_type in BaseRobustness.test_types.items() if name.startswith('image_')

        ],
    )
    def test_transform(self, robustness: BaseRobustness) -> None:
        """Test the transform method of robustness-related classes.

        Args:
            robustness (BaseRobustness): A robustness-related class to test.

        Returns:
            None
        """
        sample = VisualQASample(original_image=self.image)
        transform_results = robustness.transform(sample_list=[sample])
        assert isinstance(transform_results, list)

        for result in transform_results:
            assert isinstance(result, VisualQASample)
            assert result.category == "robustness"
            assert result.perturbed_image is not None
            assert isinstance(result.perturbed_image, Image.Image)
            assert result.perturbed_image != self.image
            # assert result.perturbed_image.size != self.image.size
            # assert result.perturbed_image.mode == self.image.mode
            # assert result.perturbed_image.info == self.image.info
            # assert result.perturbed_image.getbands() == self.image.getbands()
            # assert result.perturbed_image.getcolors() == self.image.getcolors()
            # assert result.perturbed_image.getpalette() == self.image.getpalette()