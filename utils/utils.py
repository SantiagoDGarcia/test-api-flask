import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import matplotlib.image
from typing import Any, Dict, List, Optional, Tuple
import json, imghdr
from flask import request, abort
import random

"""
import jwt
from jwt import PyJWKClient
"""
from werkzeug.utils import secure_filename


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def show(img):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    # mark_areas(layer_mask, type_bg="gray")
    plt.axis("off")
    plt.show()


def save_img_png(name, img):
    matplotlib.image.imsave("mask.png", img, cmap="gray")


def convert_numpy_to_list(array: np.ndarray):
    dumped = json.dumps(array, cls=NumpyEncoder)
    print(type(dumped))
    return dumped


def check_request(type_analisis: str, allowed_extension: List) -> None:
    # if not request.is_secure:
    # abort(407)
    if (
        request.files.get("file") is None
        or type_analisis != "ultrasound"
        and type_analisis != "mammography"
    ):
        abort(404)
    if not allowed_file(request.files["file"], allowed_extension):
        abort(400)


def allowed_file(file, allowed_extension) -> bool:
    filename = secure_filename(file.filename)
    stream = file.stream
    extension = "." in filename and filename.rsplit(".", 1)[1].lower()

    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)

    if not format:
        return False
    # format = format if format != "jpeg" else "jpg"

    if "." + extension in allowed_extension and extension == format:
        return True
    else:
        return False


def generate_id(length: int):
    def fragmentName():
        return hex(int((1 + random.random()) * 0x10000))[3:]

    cadena = ""
    for i in range(length - 1):
        cadena = fragmentName() + "-" + cadena
    return cadena + fragmentName()


def format_data_to_dict(
    extract_roi: str,
    path: str,
    data_to_eval: List[np.ndarray],
    original_image: Optional[np.ndarray] = None,
    drawed_image: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if extract_roi == "true":
        images = {
            "imgUrl": {"path": f"{path}original.png", "image": original_image},
            "imgDrawnUrl": {
                "path": f"{path}drawed_contours.png",
                "image": drawed_image,
            },
            "rois": {f"{path}roi{idx}.png": x for idx, x in enumerate(data_to_eval)},
        }
        data_doc = {
            "cantRoi": len(data_to_eval),
            "roiExtracted": True,
        }
        return images, data_doc
    else:
        images = {
            "imgUrl": {"path": f"{path}original.png", "image": data_to_eval[0]},
        }
    return images, {}


"""
def verify_app_check(token):
    if token is None:
        return None

    # Obtain the Firebase App Check Public Keys
    # Note: It is not recommended to hard code these keys as they rotate,
    # but you should cache them for up to 6 hours.
    url = "https://firebaseappcheck.googleapis.com/v1beta/jwks"

    jwks_client = PyJWKClient(url)
    signing_key = jwks_client.get_signing_key_from_jwt(token)

    header = jwt.get_unverified_header(token)
    # Ensure the token's header uses the algorithm RS256
    if header.get("alg") != "RS256":
        return None
    # Ensure the token's header has type JWT
    if header.get("typ") != "JWT":
        return None

    payload = {}
    try:
        # Verify the signature on the App Check token
        # Ensure the token is not expired
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            # Ensure the token's audience matches your project
            audience="projects/" + app.config["PROJECT_NUMBER"],
            # Ensure the token is issued by App Check
            issuer="https://firebaseappcheck.googleapis.com/"
            + app.config["PROJECT_NUMBER"],
        )
    except:
        print(f"Unable to verify the token")

    # The token's subject will be the app ID, you may optionally filter against
    # an allow list
    return payload.get("sub")
"""
