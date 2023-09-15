import cv2, config
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import matplotlib.image
from typing import Any, Dict, List, Optional, Tuple
import json, imghdr, random
from flask import request, abort, jsonify
from datetime import datetime
from werkzeug.utils import secure_filename

allowed_extension = config.UPLOAD_EXTENSIONS


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # Encode numpy data types as native Python data types for JSON serialization
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def convert_numpy_to_list(array: np.ndarray):
    # Convert numpy array to list using custom JSON encoder
    dumped = json.dumps(array, cls=NumpyEncoder)
    print(type(dumped))
    return dumped


def check_request(type_analisis: str) -> None:
    # Check if request is valid and authorized
    auth_header = request.headers.get("Authorization")
    if auth_header:
        auth_scheme, auth_token = auth_header.split()
        if auth_scheme != "Bearer":
            abort(401)
    else:
        abort(401)
    # Check if file is present and type_analisis is valid
    if (
        request.files.get("file") is None
        or type_analisis != "ultrasound"
        and type_analisis != "mammography"
    ):
        abort(401)
    # Check if file has allowed extension
    if not allowed_file(request.files["file"], allowed_extension):
        abort(generate_response("ERROR", "NO FILE ALLOWED"))


def allowed_file(file, allowed_extension) -> bool:
    # Check if file has allowed extension and format
    filename = secure_filename(file.filename)
    stream = file.stream
    extension = "." in filename and filename.rsplit(".", 1)[1].lower()
    # Check file format using imghdr module
    header = stream.read(512)
    stream.seek(0)
    format = imghdr.what(None, header)
    # Normalize format and extension values for comparison
    if not format:
        return False
    else:
        format = format if format != "jpeg" else "jpg"
        extension = extension if extension != "jpeg" else "jpg"
    # Check if extension and format are allowed and match
    if "." + extension in allowed_extension and extension == format:
        return True
    else:
        return False


def generate_id(length: int):
    # Generate a random ID of specified length using hex values
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
    # Format data as dictionary for uploading to Firestore or storage bucket
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


def get_data_routes(
    user_id: str,
    predictions,
    type_analisis: str,
):
    # Generate data for uploading results to Firestore and storage bucket
    hist_id = generate_id(3)
    collection = f"Users/{user_id}/HistResults"
    current_timestamp = int(datetime.utcnow().timestamp())
    path = f"results/{user_id}/{hist_id}/"
    data_doc = {
        "testResult": "M" if "M" in predictions else "B",
        "typeAnalysis": type_analisis[0].upper(),
        "dateAnalysis": current_timestamp,
        "roiExtracted": False,
        "isActive": True,
    }
    return hist_id, path, collection, data_doc


def generate_response(
    result: str, aditionalInfo: str = None, idResult: Optional[str] = None
):
    # Generate a JSON response with the specified result and additional information
    return jsonify(
        {
            "result": result,
            "aditionalInfo": aditionalInfo,
            "idResult": idResult,
        }
    )
