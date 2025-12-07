from flask import Flask, request, render_template
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model('extremist_detector.keras')

@app.route('/')
def home():
    return render_template('message.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('message')
    pred = float(model.predict(tf.constant([text]))[0][0])
    label = 'EXTREMIST' if pred < 0.5 else 'SAFE'

    return render_template('predict.html', label=label, score=round(pred*100, 2))

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Render assigns the port
    app.run(host='0.0.0.0', port=port) 
    
