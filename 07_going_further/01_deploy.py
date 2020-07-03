from flask import Flask, request, jsonify
import tensorflow as tf

# App
app = Flask(__name__)

# Model
model = tf.keras.models.load_model('predictor')

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    data = float(content['data'])
    outputs_predict = model.predict([data])

    return jsonify({"prediction": str(outputs_predict[0][0])})
