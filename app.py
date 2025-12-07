from flask import Flask, request, render_template
import tensorflow as tf
import os

# Reduce TensorFlow log clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 2 = warnings/errors only

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model('extremist_detector.keras')

@app.route('/')
def home():
    return render_template('message.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('message')
    # Predict using TensorFlow model
    pred = float(model.predict(tf.constant([text]))[0][0])
    label = 'EXTREMIST' if pred < 0.5 else 'SAFE'
    return render_template('predict.html', label=label, score=round(pred*100, 2))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Only 1 worker to save memory
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='0.0.0.0', port=port)
