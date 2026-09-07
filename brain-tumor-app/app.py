import os
import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from flask import Flask, request, jsonify, render_template

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'brain_tumor_model.h5')
IMG_SIZE = 224
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

app = Flask(__name__)
model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# Target layer for Grad-CAM: the last Conv2D in the model (EfficientNet top_conv)
_conv_layers = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)]
GRAD_CAM_LAYER = _conv_layers[-1].name
print("Grad-CAM layer:", GRAD_CAM_LAYER)


def preprocess_image(file_storage):
    img_bytes = file_storage.read()
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR)
    img = tf.expand_dims(img, axis=0)
    return img


def _jet_lut():
    x = np.linspace(0.0, 1.0, 256)
    r = np.interp(x, [0, 0.125, 0.375, 0.625, 0.875, 1.0], [0, 0, 0, 1, 1, 0.5])
    g = np.interp(x, [0, 0.125, 0.375, 0.625, 0.875, 1.0], [0, 0, 1, 1, 0, 0])
    b = np.interp(x, [0, 0.125, 0.375, 0.625, 0.875, 1.0], [0.5, 1, 1, 0, 0, 0])
    return np.stack([r, g, b], axis=1)


_JET = _jet_lut()


def compute_gradcam(x):
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(GRAD_CAM_LAYER).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(x)
        top_class = tf.argmax(preds[0])
        loss = preds[0][top_class]
    grads = tape.gradient(loss, conv_output)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_output[0]), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + keras.backend.epsilon())
    cam = tf.image.resize(cam[..., tf.newaxis], (IMG_SIZE, IMG_SIZE), method='bilinear')
    return cam.numpy()[..., 0]


def make_overlay(x, cam):
    heat = np.uint8(np.clip(cam * 255.0, 0, 255))
    heat_color = _JET[heat]                     # (224,224,3) float 0..1
    img = np.squeeze(x.numpy(), axis=0)         # (224,224,3) float 0..255
    img = np.clip(img, 0, 255) / 255.0
    overlay = 0.55 * img + 0.45 * heat_color
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        x = preprocess_image(file)
        preds = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        result = {
            'class_name': CLASS_LABELS[idx],
            'confidence': round(confidence * 100, 2),
            'probabilities': {k: round(float(v) * 100, 2) for k, v in zip(CLASS_LABELS, preds)},
        }
        try:
            cam = compute_gradcam(x)
            result['heatmap'] = make_overlay(x, cam)
        except Exception as e:
            print("Grad-CAM skipped:", e)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': 'Prediction failed: {}'.format(str(e))}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)