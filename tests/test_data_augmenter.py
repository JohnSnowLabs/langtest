import os
import pytest

from langtest.augmentation.augmenter import DataAugmenter


@pytest.mark.parametrize(
    "config, task, data_info, save_path",
    [
        (
            "tests/fixtures/augmenter_config.yaml",
            "ner",
            {"data_source": "tests/fixtures/test.conll"},
            "tests/fixtures/augmented_test.conll",
        ),
        (
            "tests/fixtures/augmenter_config.yaml",
            "text-classification",
            {"data_source": "tests/fixtures/text_classification.csv"},
            "tests/fixtures/augmented_text_classification.csv",
        ),
    ],
)
def test_DataAugmenter(config, task, data_info, save_path):
    # Initialize the DataAugmenter
    augmenter = DataAugmenter(task, config)

    # Load the data
    augmenter.augment(data=data_info)

    # save the augmented data
    augmenter.save(save_path)

    # Check if the save path exists
    assert os.path.exists(save_path) == True

    # Remove the saved file for cleaning
    os.remove(save_path)


def test_DataAugmenter_invalid_task():
    with pytest.raises(ValueError):
        augmenter = DataAugmenter("invalid_task", "tests/fixtures/augmenter_config.yaml")
        augmenter.augment(data={"data_source": "tests/fixtures/test.conll"})


def test_DataAugmenter_invalid_config():
    with pytest.raises(AttributeError):
        augmenter = DataAugmenter("ner", {"config": "tests/fixtures/invalid_config.yaml"})
        augmenter.augment(data={"data_source": "tests/fixtures/test.conll"})


def test_DataAugmenter_invalid_data():
    with pytest.raises(FileNotFoundError):
        augmenter = DataAugmenter("ner", "tests/fixtures/augmenter_config.yaml")
        augmenter.augment(data={"data_source": "tests/fixtures/invalid_data.conll"})
