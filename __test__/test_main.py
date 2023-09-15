import json
import unittest
from flask import Flask
from ..app import app


class FlaskTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_evaluate_image(self):
        response = self.app.post(
            "/evaluate-image/test_type_analisis/",
            data=dict(
                extract_roi="true",
                file=(open("000018.png", "rb"), "000018.png"),
                idToken="test_token",
            ),
        )
        data = json.loads(response.data)
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIsNotNone(data["data"])


if __name__ == "__main__":
    unittest.main()
