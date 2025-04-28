from flask import Flask, render_template, request, send_file, jsonify
import os
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import cv2

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["BACKGROUND_FOLDER"] = "static/backgrounds"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["BACKGROUND_FOLDER"], exist_ok=True)

# Model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.route("/")
def index():
    backgrounds = [
        f
        for f in os.listdir(app.config["BACKGROUND_FOLDER"])
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    return render_template("index.html", backgrounds=backgrounds)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)
        return jsonify(
            {"message": "File uploaded successfully", "filename": file.filename}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process", methods=["POST"])
def process_image():
    data = request.json
    filename = data.get("filename")
    action = data.get("action")
    bg_filename = data.get("bg_filename")

    if not filename:
        return jsonify({"error": "Filename not provided"}), 400

    try:
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        img = img.resize((512, 512))

        # Segmentation
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)["out"][0]
        predicted_mask = output.argmax(0)

        # Processing
        if action == "remove_bg":
            human_mask = (predicted_mask == 15).cpu().numpy()
            result = apply_mask(np.array(img), human_mask)
        elif action == "change_bg" and bg_filename:
            bg_path = os.path.join(app.config["BACKGROUND_FOLDER"], bg_filename)
            new_bg = Image.open(bg_path).convert("RGB").resize((512, 512))
            human_mask = (predicted_mask == 15).cpu().numpy()
            result = change_background(np.array(img), np.array(new_bg), human_mask)
        else:
            return jsonify({"error": "Invalid operation"}), 400

        # Save result
        # Sonucu kaydet
        result_filename = f"processed_{filename.split('.')[0]}.png"  # JPEG yerine PNG
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)

        # PNG olarak kaydet
        Image.fromarray(result).resize(original_size).save(result_path, format="PNG")

        return jsonify(
            {"message": "Operation successful", "result_filename": result_filename}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    # Dosya uzantısını kontrol et
    if filename.lower().endswith(".png"):
        mimetype = "image/png"
    else:
        mimetype = "image/jpeg"

    return send_file(file_path, as_attachment=True, mimetype=mimetype)


def apply_mask(image, mask):
    rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask * 255
    return rgba


def change_background(image, new_bg, mask):
    mask_3d = np.stack([mask] * 3, axis=-1)
    return (image * mask_3d + new_bg * (1 - mask_3d)).astype(np.uint8)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
