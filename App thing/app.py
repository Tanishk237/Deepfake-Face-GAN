import io
import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from PIL import ImageEnhance

app = Flask(__name__)
CORS(app)

# Get the absolute path to the model
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "Models", "trained_model.h5")

print(f"Loading model from: {model_path}")
generator = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")


# Function to generate an AI image
def generate_image(seed_value):
    noise_dim = 100
    np.random.seed(seed_value)
    random_noise = np.random.randn(1, noise_dim)
    generated_image = generator.predict(random_noise, verbose=0)

    # Normalize and convert to uint8
    generated_image = (generated_image + 1) / 2
    generated_image_uint8 = (generated_image[0] * 255).astype(np.uint8)

    # TensorFlow outputs RGB, so we keep it in RGB throughout
    # First, upscale to higher resolution using LANCZOS4 (best quality)
    upscaled_image = cv2.resize(generated_image_uint8, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply bilateral filter to reduce noise while preserving edges
    # Note: Keep in RGB space (don't convert to BGR)
    upscaled_image = cv2.bilateralFilter(upscaled_image, 5, 50, 50)
    
    # Apply unsharp masking for sharpening
    gaussian = cv2.GaussianBlur(upscaled_image, (5, 5), 2.0)
    upscaled_image = cv2.addWeighted(upscaled_image, 1.5, gaussian, -0.5, 0)
    
    # Ensure values are in valid range [0, 255]
    upscaled_image = np.clip(upscaled_image, 0, 255).astype(np.uint8)
    
    # Convert to PIL Image (ensure RGB mode)
    img = Image.fromarray(upscaled_image, 'RGB')
    
    # Optional: Enhance contrast slightly
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    img_io = io.BytesIO()
    img.save(img_io, format="PNG")
    img_io.seek(0)

    return img_io


@app.route('/')
def home():
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.route('/generate', methods=['GET'])
def generate():
    seed_value = int(request.args.get("seed", np.random.randint(10000)))
    print(f"Generating image for seed: {seed_value}")

    img_io = generate_image(seed_value)
    print("Sending image response...")

    response = Response(img_io.getvalue(), mimetype="image/png")
    response.headers["Cache-Control"] = "no-store"

    return response


if __name__ == "__main__":
    app.run(debug=True)
