from flask import Flask, jsonify, request, abort
from utils.create_mask import ExtractMasks
from utils.eval_images import Predict
from utils.utils import (
    show,
    convert_numpy_to_list,
    check_request,
    generate_id,
    format_data_to_dict,
)
import warnings, json, time, uuid
from utils.firebase_functions import FirebaseManager
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10024 * 10024
app.config["UPLOAD_EXTENSIONS"] = [".jpg", ".png", ".jpeg"]

# Crear una instancia de FirebaseManager
firebase_manager = FirebaseManager()


# Define the route /evaluate-image/<string:type_analisis>/ that accepts POST requests
@app.route("/", methods=["POST"])
def index():
    return "<h2>TEST OK</h2>"

@app.route("/evaluate-image/<string:type_analisis>/", methods=["POST"])
def evaluate_image(type_analisis: str):
    check_request(type_analisis, app.config["UPLOAD_EXTENSIONS"])
    user_id = request.form["userId"]
    extract_roi = str(request.args.get("extract_roi"))
    filestr = request.files["file"].read()
    aditional_info = None

    if extract_roi == "true":
        original_image, drawed_image, data_to_eval = ExtractMasks(
            img_input_file=filestr
        ).get_masks()
        if len(data_to_eval) == 0:
            return jsonify(
                {
                    "result": "ERROR",
                    "aditionalInfo": "NO ROI FOUND",
                }
            )
    else:
        original_image = drawed_image = None
        data_to_eval = [
            ExtractMasks(img_input_file=filestr).image_to_numpy_array(
                transform_gray=True
            )
        ]

    predictions = Predict(
        list_masks=data_to_eval, type_analisis=type_analisis
    ).get_predictions()

    histId = generate_id(3)
    collection = f"Users/{user_id}/HistResults"
    document = histId
    current_timestamp = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    path = f"results/{user_id}/{histId}/"

    data_doc = {
        "testResult": "M" if "M" in predictions else "B",
        "typeAnalysis": type_analisis[0].upper(),
        "dateAnalysis": current_timestamp,
        "roiExtracted": False,
    }

    images, extra_data = format_data_to_dict(
        extract_roi, path, data_to_eval, original_image, drawed_image
    )
    data_doc.update(extra_data)

    result, aditional_info = firebase_manager.upload_results(
        images, collection, document, data_doc
    )
    print(f"Resultado: {result}")

    json_response = {
        "result": result,
        "aditionalInfo": aditional_info,
        "idResult": histId,
    }
    return jsonify(json_response)


if __name__ == "__main__":
    app.run(
        debug=True,
        # ssl_context=("newcertificate.crt", "newkey.key"),
    )
