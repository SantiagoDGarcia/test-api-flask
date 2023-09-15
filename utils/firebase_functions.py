import firebase_admin
from firebase_admin import credentials, firestore, storage, auth
import os, time
from PIL import Image
import io
import numpy as np
import datetime
import config


class FirebaseManager:
    def __init__(self):
        # Initialize Firebase app with credentials and storage provider
        cred = credentials.Certificate(config.firebase_config)
        firebase_admin.initialize_app(
            cred, {"storageBucket": config.firebase_storage_provider}
        )
        self.db = firestore.client()
        self.storage_client = storage.bucket()

    def upload_image(self, path: str, image: Image) -> str:
        # Save image to memory buffer in PNG format
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            image_data = output.getvalue()
        # Upload image data to storage bucket at specified path
        blob = self.storage_client.blob(path)
        blob.upload_from_string(image_data, content_type="image/png")
        # Make image public and get public URL
        blob.make_public()
        url = blob.public_url
        return url

    def upload_results(
        self,
        start_time: float,
        images: dict,
        collection: str,
        document: str,
        data: dict,
    ):
        uploaded_files = []
        uploaded_urls = {}

        try:
            # Upload images and store URLs in uploaded_urls dictionary
            for key, image_dict in images.items():
                if key != "rois":
                    path = image_dict["path"]
                    image = Image.fromarray(image_dict["image"])
                    url = self.upload_image(path, image)
                    uploaded_files.append(path)
                    uploaded_urls[key] = url
                else:
                    uploaded_urls["imgRoiUrl"] = []
                    for path, image in images["rois"].items():
                        image = Image.fromarray(image)
                        url = self.upload_image(path, image)
                        uploaded_files.append(path)
                        uploaded_urls["imgRoiUrl"].append(url)

            # Add uploaded URLs to data dictionary
            for key, description in uploaded_urls.items():
                data[key] = description
            # Calculate elapsed time and add to data dictionary
            end_time = time.time()
            elapsed_time = round((end_time - start_time) / 60, 2)
            data["durationAnalysis"] = elapsed_time
            # Add document to Firestore collection with data
            self.add_document(collection=collection, document=document, data=data)
            return "OK", None

        except Exception as e:
            # If an error occurs, delete all uploaded files and document from Firestore
            for path in uploaded_files:
                blob = self.storage_client.blob(path)
                blob.delete()
            self.delete_document(collection=collection, document=document)
            print(f"Error: {e}")
            return "ERROR", "NO DATA UPLOAD"

    def add_document(self, collection: str, document: str, data: dict):
        # Add a document to a Firestore collection with the specified data
        doc_ref = self.db.collection(collection).document(document)
        doc_ref.set(data)

    def delete_document(self, collection: str, document: str):
        # Delete a document from a Firestore collection
        self.db.collection(collection).document(document).delete()

    def check_token(self, id_token: str):
        # Verify Firebase ID token and return user ID if valid
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token["uid"]
            return uid
        except Exception as error:
            print("Error: ", error)
            return None
