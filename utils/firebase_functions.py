import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from dotenv import load_dotenv
from PIL import Image
import io
import numpy as np
import datetime


class FirebaseManager:
    def __init__(self):
        load_dotenv()

        firebase_config = {
            "type": os.getenv("FIREBASE_TYPE"),
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
            "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv(
                "FIREBASE_AUTH_PROVIDER_X509_CERT_URL"
            ),
            "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
            "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN"),
        }

        cred = credentials.Certificate(firebase_config)

        firebase_admin.initialize_app(
            cred, {"storageBucket": os.getenv("FIREBASE_STORAGE_PROVIDER")}
        )
        self.db = firestore.client()
        self.storage_client = storage.bucket()

    def upload_image(self, path: str, image: Image) -> str:
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            image_data = output.getvalue()

        blob = self.storage_client.blob(path)
        blob.upload_from_string(image_data, content_type="image/png")

        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(minutes=15),
            method="GET",
        )
        return url

    def upload_results(self, images: dict, collection: str, document: str, data: dict):
        uploaded_files = []
        uploaded_urls = {}
        uploaded_urls["imgRoiUrl"] = []
        try:
            for key, image_dict in images.items():
                if key != "rois":
                    path = image_dict["path"]
                    image = Image.fromarray(image_dict["image"])
                    url = self.upload_image(path, image)
                    uploaded_files.append(path)
                    uploaded_urls[key] = url
                else:
                    for path, image in images["rois"].items():
                        image = Image.fromarray(image)
                        url = self.upload_image(path, image)
                        uploaded_files.append(path)
                        uploaded_urls["imgRoiUrl"].append(url)
            for key, description in uploaded_urls.items():
                data[key] = uploaded_urls[key]

            self.add_document(collection=collection, document=document, data=data)

            return "OK", None
        except Exception as e:
            for path in uploaded_files:
                blob = self.storage_client.blob(path)
                blob.delete()

            self.delete_document(collection=collection, document=document)
            return f"Error: {e}"

    def add_document(self, collection: str, document: str, data: dict):
        doc_ref = self.db.collection(collection).document(document)
        doc_ref.set(data)

    def delete_document(self, collection: str, document: str):
        self.db.collection(collection).document(document).delete()
