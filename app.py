import os
import requests
import io
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
from transformers import pipeline
from huggingface_hub import InferenceClient
from music_genrator import MusicGenerator 
import google.generativeai as genai# Import your music generator class

app = Flask(__name__)



# Replace 'YOUR_API_KEY' with your actual API key
genai.configure(api_key="AIzaSyDHAfBkaEK3BrUKL04UKz-9KoCZDnZSNcg")

# Select the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

# Prompt for text generation


# Folder to save generated music files
UPLOAD_FOLDER = 'generated_music'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Hugging Face API configuration for image generation
API_URL = "https://api-inference.huggingface.co/models/Yntec/HyperRealism"
headers = {"Authorization": "Bearer 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"}

# Initialize the text generation pipeline for literature generation
pipe = pipeline("text-generation", model="gpt2")  # Use "gpt2" or a similar model


# Function to query Hugging Face API for image generation
def query_huggingface(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

# Route for the Image Generator page
@app.route('/image-generator', methods=['GET', 'POST'])
def image_generator():
    if request.method == 'POST':
        description = request.form.get('description')
        style = request.form.get('style')
        resolution = request.form.get('resolution')

        # Prepare the input prompt based on user input
        prompt = f"{description} in {style} style"

        # Query the Hugging Face API
        image_bytes = query_huggingface({
            "inputs": prompt
        })

        # Check if the API response contains valid image data
        try:
            image = Image.open(io.BytesIO(image_bytes))
            img_io = io.BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            
            # Encode the image as base64 for displaying it in the HTML
            image_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
            image_url = f"data:image/png;base64,{image_data}"

            # Render the template with the generated image
            return render_template('image-generator.html', image_url=image_url)
        
        except UnidentifiedImageError:
            return render_template('image-generator.html', error="Unable to generate an image. Please try again.")
    
    # Render the form if GET request
    return render_template('image-generator.html')

# Home Route
@app.route('/') 
def home():
    return render_template('index.html')

# Music Generator Page Route
@app.route('/music-generator', methods=['GET', 'POST'])
def music_generator_page():
    generated_file = None

    if request.method == 'POST':
        # Get the prompt or music description from the user
        prompt = request.form.get('prompt')

        # Initialize the MusicGenerator
        music_generator = MusicGenerator()
        
        # Generate music and save it to the 'generated_music' folder
        result = music_generator.generate_music(prompt)
        filename = result['file']
        generated_file = f"/generated_music/{filename}"

        # Return the page with the download link
        return render_template('music-generator.html', generated_file="generated_music")
    
    return render_template('music-generator.html', generated_file="generated_music")

# Literature Generator Page Route
@app.route('/literature-generator', methods=['GET', 'POST'])
def literature_generator():
    generated_text = None

    if request.method == 'POST':
        writing_type = request.form.get('type')
        theme = request.form.get('theme')
        length = int(request.form.get('length'))

        prompt = f"Write a {writing_type} about {theme} with {length} words in a single line."

        response = model.generate_content(prompt)
        generated_text = response.text

        # You can return the generated text directly as a string:
        return render_template('literature-generator.html', generated_text=generated_text)

        # Or you can return it as JSON:
        #return jsonify({'generated_text': generated_text})

    return render_template('literature-generator.html', generated_text=generated_text)

# About Page Route
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Page Route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Serve generated music file
@app.route('/generated_music/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Make sure the generated_music folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
