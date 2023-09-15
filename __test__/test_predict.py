import unittest
import numpy as np
from ..utils.eval_images import Predict


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.predict = Predict(
            list_masks=np.array([[[0, 0], [0, 0]]]), type_analisis="test_type_analisis"
        )

    def test_transform_numpy_to_tensor(self):
        list_masks = np.array([[[0, 0], [0, 0]]])
        result = self.predict.transform_numpy_to_tensor(list_masks)
        self.assertIsNotNone(result)

    def test_make_predictions(self):
        model_net = None
        image = np.array([[0, 0], [0, 0]])
        model_dir = "test_model_dir"
        result = self.predict.make_predictions(model_net, image, model_dir)
        self.assertIsNone(result)

    def test_get_predictions(self):
        result = self.predict.get_predictions()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
