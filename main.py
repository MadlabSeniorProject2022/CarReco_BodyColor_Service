from flask import Flask, request, jsonify
import predict
import cv2
import numpy as np
import io
import os

app = Flask(__name__)

@app.route("/useColor", methods=["POST"])
def color_predict():
    if request.method != "POST":
        return jsonify({'msg': 'notfound', 'predicted': None})

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        file_bytes = np.asarray(bytearray(io.BytesIO(im_bytes).read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = predict.knn_predict(img)

    return jsonify({'msg': 'success', 'predicted': result})

@app.route("/raw_color", methods=["POST"])
def raw_color_predict():
    if request.method != "POST":
        return jsonify({'msg': 'notfound', 'predicted': None})

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        file_bytes = np.asarray(bytearray(io.BytesIO(im_bytes).read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        result = predict.raw_predict(img)

    return jsonify({'msg': 'success', 'predicted': result})
@app.route("/")
def ping():
    return "Hello, This is service for Car Color Classification"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))