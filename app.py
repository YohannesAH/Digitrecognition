import numpy as np
import cv2
import keras.models as models
from flask import Flask, request, render_template, url_for, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

model = models.load_model('mymodel')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_digit():
    # Read the image using OpenCV
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28)) # Resize - important!
    img = (img / 255) - 0.5  # Normalize image data
    img_tensor = np.expand_dims(img, axis=0)
    prediction = model.predict(img_tensor)
    predicted_digit = np.argmax(prediction[0])
    # Pass the predicted digit and uploaded image to the template
    return render_template('index.html', digit=int(predicted_digit), file=request.files['image'])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
