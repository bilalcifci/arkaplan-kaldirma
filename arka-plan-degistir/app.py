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

# Model tanimi
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet101(pretrained=True).to(cihaz)
model.eval()

donusum = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.route("/")
def anasayfa():
    arkaplanlar = [
        f
        for f in os.listdir(app.config["BACKGROUND_FOLDER"])
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    return render_template("index.html", backgrounds=arkaplanlar)


@app.route("/upload", methods=["POST"])
def dosya_yukle():
    if "file" not in request.files:
        return jsonify({"error": "Dosya yuklenmedi"}), 400

    dosya = request.files["file"]
    if dosya.filename == "":
        return jsonify({"error": "Dosya secilmedi"}), 400

    try:
        dosya_adi = os.path.join(app.config["UPLOAD_FOLDER"], dosya.filename)
        dosya.save(dosya_adi)
        return jsonify(
            {"message": "Dosya basariyla yuklendi", "filename": dosya.filename}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process", methods=["POST"])
def isleme():
    veri = request.json
    dosya_adi = veri.get("filename")
    islem = veri.get("action")
    arkaplan_adi = veri.get("bg_filename")

    if not dosya_adi:
        return jsonify({"error": "Dosya adi verilmedi"}), 400

    try:
        yol = os.path.join(app.config["UPLOAD_FOLDER"], dosya_adi)
        gorsel = Image.open(yol).convert("RGB")
        orjinal_boyut = gorsel.size
        gorsel = gorsel.resize((512, 512))

        # Segmentasyon
        giris_tensor = donusum(gorsel).unsqueeze(0).to(cihaz)
        with torch.no_grad():
            cikti = model(giris_tensor)["out"][0]
        tahmin_maskesi = cikti.argmax(0)

        # Islem secimi
        if islem == "remove_bg":
            insan_maskesi = (tahmin_maskesi == 15).cpu().numpy()
            sonuc = maske_uygula(np.array(gorsel), insan_maskesi)
        elif islem == "change_bg" and arkaplan_adi:
            yeni_arkaplan_yol = os.path.join(
                app.config["BACKGROUND_FOLDER"], arkaplan_adi
            )
            yeni_arkaplan = (
                Image.open(yeni_arkaplan_yol).convert("RGB").resize((512, 512))
            )
            insan_maskesi = (tahmin_maskesi == 15).cpu().numpy()
            sonuc = arkaplan_degistir(
                np.array(gorsel), np.array(yeni_arkaplan), insan_maskesi
            )
        else:
            return jsonify({"error": "Gecersiz islem"}), 400

        # Sonucu kaydet
        sonuc_adi = f"processed_{dosya_adi.split('.')[0]}.png"
        sonuc_yolu = os.path.join(app.config["UPLOAD_FOLDER"], sonuc_adi)

        Image.fromarray(sonuc).resize(orjinal_boyut).save(sonuc_yolu, format="PNG")

        return jsonify({"message": "Islem basarili", "result_filename": sonuc_adi})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def indir(filename):
    dosya_yolu = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if filename.lower().endswith(".png"):
        mimetype = "image/png"
    else:
        mimetype = "image/jpeg"

    return send_file(dosya_yolu, as_attachment=True, mimetype=mimetype)


def maske_uygula(gorsel, maske):
    rgba = cv2.cvtColor(gorsel, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = maske * 255
    return rgba


def arkaplan_degistir(gorsel, yeni_arkaplan, maske):
    maske_3d = np.stack([maske] * 3, axis=-1)
    return (gorsel * maske_3d + yeni_arkaplan * (1 - maske_3d)).astype(np.uint8)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
