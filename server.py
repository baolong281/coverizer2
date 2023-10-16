from flask import Flask, request, send_from_directory
from PIL import Image
from predict import Predictor
import base64
import io
import json
from werkzeug.utils import secure_filename
import secrets
import os
import atexit
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "stored_images"

primer  = Predictor()

def generate_random_filename():
    random_hex = secrets.token_hex(16)
    return f"{random_hex}.png"

if os.path.exists(app.config["UPLOAD_FOLDER"]) == False:
    os.mkdir(app.config["UPLOAD_FOLDER"])

@app.route('/api/inference', methods=['POST'])
def inference():
    print("running inference")
    req_data = request.get_json()
    image_binary = req_data['image'].split(',')[1]
    res = int(req_data['res'])
    image_binary  = base64.b64decode(image_binary)
    image = Image.open(io.BytesIO(image_binary))
    out, _, _ = primer.predict(img=image, tosize=(res, res), size=0.8)

    fp = os.path.join(app.config["UPLOAD_FOLDER"], generate_random_filename()) 
    out.save(fp)
    fp = f"/api/image/{os.path.basename(fp)}"

    return json.dumps({"filename": fp})

@app.route('/api/image/<filename>', methods=['GET'])
def temporary_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/api/test')
def test():
    return "<p>Hello, World!</p>"

# Add a function to clean up temporary files when the app exits
def cleanup_temporary_files():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.remove(file_path)

atexit.register(cleanup_temporary_files)
