import unittest
import numpy as np
from ..utils.create_mask import ExtractMasks


class TestExtractMasks(unittest.TestCase):
    def setUp(self):
        self.extract_masks = ExtractMasks(img_input_file="000018.png")

    def test_mark_areas(self):
        anns = np.array([{"segmentation": np.array([[0, 0, 1, 1]]), "area": 1}])
        result = self.extract_masks.mark_areas(anns)
        self.assertIsNotNone(result)

    def test_pad_images(self):
        img = np.array([[0, 0], [0, 0]])
        result = self.extract_masks.pad_images(img, 1, 1)
        self.assertEqual(result.shape, (4, 4))

    def test_extract_roi_images(self):
        img = np.array([[0, 0], [0, 0]])
        mask = np.array([[0, 0], [0, 0]])
        result = self.extract_masks.extract_roi_images(img, mask)
        self.assertIsNotNone(result)

    def test_image_to_numpy_array(self):
        result = self.extract_masks.image_to_numpy_array()
        self.assertIsNotNone(result)

    def test_get_masks(self):
        result = self.extract_masks.get_masks()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
