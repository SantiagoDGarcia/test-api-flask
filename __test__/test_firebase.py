import unittest
from ..utils.firebase_functions import FirebaseManager
from PIL import Image

firebase_manager = FirebaseManager()


class TestFirebaseManager(unittest.TestCase):
    def test_upload_image(self):
        path = "test_path"
        image = Image.new("RGB", (1, 1))
        result = firebase_manager.upload_image(path, image)
        self.assertIsNotNone(result)

    def test_upload_results(self):
        start_time = 0
        images = {}
        collection = "test_collection"
        document = "test_document"
        data = {}
        result = firebase_manager.upload_results(
            start_time, images, collection, document, data
        )
        self.assertIsNotNone(result)

    def test_add_document(self):
        collection = "test_collection"
        document = "test_document"
        data = {}
        result = firebase_manager.add_document(collection, document, data)
        self.assertIsNone(result)

    def test_delete_document(self):
        collection = "test_collection"
        document = "test_document"
        result = firebase_manager.delete_document(collection, document)
        self.assertIsNone(result)

    def test_check_token(self):
        id_token = "test_token"
        result = firebase_manager.check_token(id_token)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
