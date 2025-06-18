from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Initialize the Flask application
app = Flask(__name__)

# Load the best saved model
model = load_model('./saved_models/best_model.h5')

# Define image size for preprocessing
IMG_SIZE = (224, 224)

# Define the categories
CATEGORIES = ['Normal', 'Pneumonia']

# Define the directory for uploaded images
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set upload folder configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        
        # Predict the category
        pred = model.predict(img_array)
        pred_class = np.argmax(pred, axis=1)
        result = CATEGORIES[pred_class[0]]
        
        # Return the result page with the prediction
        return render_template('result.html', result=result, filename=file.filename)

# Serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
