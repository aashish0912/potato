import requests
import io
import PIL
from PIL import Image
from flask import Flask, render_template, request, jsonify
import base64
from transformers import pipeline
from huggingface_hub import InferenceClient
import google.generativeai as genai

app = Flask(__name__)

# Initialize the text generation pipeline
 # Or any other model you prefer

# Initialize the Hugging Face InferenceClient for the chat model
genai.configure(api_key="AIzaSyDzP1t1R-qJ89kKpx0F55ilJzeVhycRMzQ")

# Select the Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")


# Hugging Face API configuration for image generation
API_URL = "https://api-inference.huggingface.co/models/Yntec/HyperRealism"
headers = {"Authorization": "Bearer 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"}

# Function to generate literature
def generate_literature(writing_type, description, length):
    prompt = f"Generate a {writing_type} based on the following description: {description}"
    result = pipe(prompt, max_length=length, num_return_sequences=1)
    return result[0]['generated_text']

# Function to query the Hugging Face API for image generation
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
        
        except PIL.UnidentifiedImageError:
            return render_template('image-generator.html', error="Unable to generate an image. Please try again.")
    
    # Render the form if GET request
    return render_template('image-generator.html')

# Home Route
@app.route('/') 
def home():
    return render_template('index.html')

# Music Generator Page Route
@app.route('/music-generator')
def music_generator():
    return render_template('music-generator.html')

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

if __name__ == '__main__':
    app.run(debug=True)
