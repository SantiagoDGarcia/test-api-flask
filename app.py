import json, time, uuid, warnings, config
from flask_cors import CORS, cross_origin
from datetime import datetime
from flask import Flask, jsonify, request, abort
from utils.create_mask import ExtractMasks
from utils.eval_images import Predict
from utils.firebase_functions import FirebaseManager
from utils.utils import (
    check_request,
    convert_numpy_to_list,
    format_data_to_dict,
    get_data_routes,
    generate_response,
)

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
app.config["UPLOAD_EXTENSIONS"] = config.UPLOAD_EXTENSIONS
CORS(
    app,
    origins=["*"],
    methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=["10"],
)

# Create an instance of FirebaseManager
firebase_manager = FirebaseManager()


# Define the route /evaluate-image/<string:type_analisis>/ that accepts POST requests
@app.route("/evaluate-image/<string:type_analisis>/", methods=["POST"])
@cross_origin()
def evaluate_image(type_analisis: str):
    # Check the request for valid file type and type_analisis
    check_request(type_analisis)

    auth_header = request.headers.get("Authorization")
    auth_scheme, auth_token = auth_header.split()
    start_time = time.time()
    # Get the parameters
    extract_roi = str(request.args.get("extract_roi"))
    filestr = request.files["file"].read()
    user_id = firebase_manager.check_token(auth_token)

    # If no user ID is found, return an error response
    if user_id is None:
        return generate_response("ERROR", "NO TOKEN ID VERIFIED")
    # If extract_roi is true, extract masks from the image
    if extract_roi == "true":
        original_image, drawed_image, data_to_eval = ExtractMasks(
            img_input_file=filestr
        ).get_masks()
        # If no data is found, return an error response
        if len(data_to_eval) == 0:
            return generate_response("ERROR", "NO ROI FOUND")
    # If extract_roi is not true, set original_image and drawed_image to None and convert the image to a numpy array
    else:
        original_image = drawed_image = None
        data_to_eval = [
            ExtractMasks(img_input_file=filestr).image_to_numpy_array(
                transform_gray=True
            )
        ]
    # Get predictions for the data using the Predict class
    predictions = Predict(
        list_masks=data_to_eval, type_analisis=type_analisis
    ).get_predictions()
    # Get data routes and images for uploading to Firebase
    hist_id, path, collection, data_doc = get_data_routes(
        user_id, predictions, type_analisis
    )
    images, extra_data = format_data_to_dict(
        extract_roi, path, data_to_eval, original_image, drawed_image
    )
    data_doc.update(extra_data)
    # Upload results to Firebase and get result and additional information
    result, aditional_info = firebase_manager.upload_results(
        start_time, images, collection, hist_id, data_doc
    )
    # Return a response with the result and additional information
    return generate_response(result, aditional_info, hist_id)


if __name__ == "__main__":
    app.run(host="192.168.1.6", port=8000, debug=True)
