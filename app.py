from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import base64

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/disease_model.keras")

# Get class names from dataset folders
class_names = sorted(os.listdir("dataset/train"))


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    uploaded_image = request.files.get("image")

    if not uploaded_image:
        return "Please select an image."

    # Save uploaded image
    image_path = "uploaded_leaf.jpg"
    uploaded_image.save(image_path)

    # Prepare image
    img = image.load_img(
        image_path,
        target_size=(128, 128)
    )

    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction) * 100)

    # Convert image to Base64 so it can be displayed
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    # Disease information
    disease_info = {

        "Early_blight": {
            "symptoms": "Dark brown spots with concentric rings appear on older leaves. Leaves may turn yellow and fall early.",
            "treatment": "Remove infected leaves and use a suitable fungicide according to agricultural recommendations.",
            "prevention": "Avoid overhead watering, maintain proper spacing, remove plant debris, and maintain good field hygiene."
        },

        "Late_blight": {
            "symptoms": "Dark water-soaked spots may appear on leaves. The affected areas can enlarge rapidly under cool and humid conditions.",
            "treatment": "Remove severely infected leaves and use an appropriate fungicide as recommended for the crop.",
            "prevention": "Avoid excess moisture, provide good air circulation, avoid overhead irrigation, and remove infected plant material."
        },

        "healthy": {
            "symptoms": "No major visible disease symptoms were detected. The leaf appears healthy.",
            "treatment": "No disease treatment is required. Continue normal crop care and monitoring.",
            "prevention": "Maintain proper irrigation, nutrition, field hygiene, and regularly inspect plants for early symptoms."
        }
    }

    # Get information
    info = disease_info.get(
        predicted_class,
        {
            "symptoms": "Disease symptoms depend on the detected crop disease.",
            "treatment": "Remove severely affected plant material and consult an agriculture expert for suitable treatment.",
            "prevention": "Maintain good crop hygiene, proper irrigation, and regular monitoring."
        }
    )

    # Display name
    display_name = predicted_class.replace("_", " ")

    # ---------------- RESULT PAGE ----------------
    return f"""
<!DOCTYPE html>

<html>
<head>

    <title>AgriCrop - Detection Result</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
            color: #263238;
        }}

        .header {{
            background: #16823b;
            color: white;
            padding: 22px;
            text-align: center;
            box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        }}

        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}

        .header p {{
            margin: 8px 0 0;
            font-size: 15px;
        }}

        .container {{
            max-width: 900px;
            margin: 35px auto;
            padding: 20px;
        }}

        .result-card {{
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        }}

        .title {{
            text-align: center;
            color: #16823b;
            margin-bottom: 25px;
        }}

        .image-box {{
            text-align: center;
            margin-bottom: 25px;
        }}

        .leaf-image {{
            width: 280px;
            height: 280px;
            object-fit: cover;
            border-radius: 15px;
            border: 5px solid #e8f5e9;
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        }}

        .disease-box {{
            text-align: center;
            background: #f1f8e9;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
        }}

        .disease-box h2 {{
            margin: 5px 0 12px;
            color: #c62828;
            font-size: 28px;
        }}

        .confidence {{
            font-size: 18px;
            font-weight: bold;
            color: #16823b;
        }}

        .progress {{
            width: 100%;
            height: 14px;
            background: #ddd;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 12px;
        }}

        .progress-bar {{
            width: {confidence:.2f}%;
            height: 100%;
            background: #16823b;
            border-radius: 10px;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
            margin-top: 20px;
        }}

        .info-card {{
            padding: 20px;
            border-radius: 15px;
            background: #fafafa;
            border-left: 5px solid #16823b;
            box-shadow: 0 3px 10px rgba(0,0,0,0.07);
        }}

        .info-card h3 {{
            margin-top: 0;
            color: #16823b;
        }}

        .info-card p {{
            line-height: 1.6;
            font-size: 14px;
        }}

        .button-box {{
            text-align: center;
            margin-top: 30px;
        }}

        .button {{
            display: inline-block;
            background: #16823b;
            color: white;
            padding: 14px 28px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: bold;
            transition: 0.3s;
        }}

        .button:hover {{
            background: #0d642d;
            transform: translateY(-2px);
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #607d63;
            font-size: 13px;
        }}

        @media(max-width: 700px) {{

            .container {{
                margin: 15px auto;
                padding: 12px;
            }}

            .result-card {{
                padding: 20px;
            }}

            .info-grid {{
                grid-template-columns: 1fr;
            }}

            .leaf-image {{
                width: 220px;
                height: 220px;
            }}

            .header h1 {{
                font-size: 26px;
            }}

        }}

    </style>

</head>

<body>

    <div class="header">
        <h1>🌱 AgriCrop</h1>
        <p>AI-Powered Crop Disease Detection</p>
    </div>


    <div class="container">

        <div class="result-card">

            <h1 class="title">
                Crop Disease Detection Result
            </h1>


            <div class="image-box">

                <img
                    class="leaf-image"
                    src="data:image/jpeg;base64,{image_data}"
                    alt="Uploaded Crop Leaf"
                >

            </div>


            <div class="disease-box">

                <div>Predicted Disease</div>

                <h2>
                    {display_name}
                </h2>

                <div class="confidence">
                    Confidence: {confidence:.2f}%
                </div>

                <div class="progress">
                    <div class="progress-bar"></div>
                </div>

            </div>


            <div class="info-grid">

                <div class="info-card">

                    <h3>🔍 Symptoms</h3>

                    <p>
                        {info["symptoms"]}
                    </p>

                </div>


                <div class="info-card">

                    <h3>💊 Treatment</h3>

                    <p>
                        {info["treatment"]}
                    </p>

                </div>


                <div class="info-card">

                    <h3>🛡️ Prevention</h3>

                    <p>
                        {info["prevention"]}
                    </p>

                </div>

            </div>


            <div class="button-box">

                <a class="button" href="/">
                    ← Check Another Leaf
                </a>

            </div>

        </div>

    </div>


    <div class="footer">
        AgriCrop © 2026 | Crop Disease Detection System
    </div>

</body>

</html>
"""


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)