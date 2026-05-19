# app.py
from flask import Flask, render_template, request, jsonify
import json
import requests
import base64
import io 
import traceback 
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION ---

# OPTION 1 (Recommended): Use .env file
# Ensure your .env file has this line: GEMINI_API_KEY=AIzaSy...
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# OPTION 2 (Quick Test): Hardcode it (Uncomment below and paste key if .env fails)
# GEMINI_API_KEY = "AIzaSyCr1xXomlmGanXkCbSJE3WxhNXGrY0z3bY" 

if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GEMINI_API_KEY not found.")
    print("1. Create a .env file in this folder.")
    print("2. Add this line: GEMINI_API_KEY=your_actual_key_here")

# --- HELPER FUNCTIONS ---

def read_pdf_text(file_content_bytes):
    """Extracts text from PDF bytes."""
    try:
        reader = PdfReader(io.BytesIO(file_content_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        print(f"Successfully extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def read_docx_text(file_content_bytes):
    """Extracts text from DOCX bytes."""
    try:
        document = Document(io.BytesIO(file_content_bytes))
        text = ""
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return None

# --- GEMINI API FUNCTIONS ---

def call_gemini_api(prompt, model="gemini-2.5-flash"): # Changed to 2.5-flash for stability
    """Calls the Gemini API for text generation."""
    if not GEMINI_API_KEY:
        return "Error: API Key is missing on server."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    
    print(f"Calling Gemini API with model: {model}...")
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        # Check for non-200 status codes (e.g., 400, 403, 500)
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            return f"Error from AI provider: {response.status_code}"

        result = response.json()
        if result.get('candidates') and result['candidates'][0].get('content'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "The AI returned no content. Please try a different prompt."
            
    except Exception as e:
        print(f"Gemini API Exception: {e}")
        traceback.print_exc()
        return "System error: Could not contact AI service."

def analyze_image_with_gemini(base64_image, mime_type):
    """Analyzes an image using Gemini Vision."""
    # Using gemini-2.5-flash which is multimodal and stable
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [
                {"text": "Analyze this medical image and provide:\n1. **Description**\n2. **Findings**\n3. **Concerns**\n4. **Recommendations**"},
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": base64_image
                    }
                }
            ]
        }]
    }
    
    print(f"Calling Gemini Vision API...")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"Vision API Error {response.status_code}: {response.text}")
            return f"Error processing image: {response.status_code}"

        result = response.json()
        if result.get('candidates') and result['candidates'][0].get('content'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "Could not analyze the image."
            
    except Exception as e:
        print(f"Vision Exception: {e}")
        return "System error during image analysis."

def analyze_text_content(text_content, file_name=""):
    """Analyzes general text content."""
    prompt = f"""Analyze the following medical report/text from '{file_name}':
    1. **Summary**
    2. **Key Metrics**
    3. **Potential Concerns**
    4. **Recommendations**

    Content: {text_content[:30000]}""" # Limit chars to avoid payload errors
    return call_gemini_api(prompt)

# --- FLASK ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    user_message = data.get('message')
    file_data_b64 = data.get('fileData')
    file_type = data.get('fileType')
    file_name = data.get('fileName')
    
    bot_response = ""
    
    try:
        if file_data_b64 and file_type:
            print(f"Processing file: {file_name}")
            file_content_bytes = base64.b64decode(file_data_b64)
            
            if file_type.startswith('image/'):
                bot_response = analyze_image_with_gemini(file_data_b64, file_type)
            
            elif file_type == 'application/pdf':
                extracted_text = read_pdf_text(file_content_bytes)
                if extracted_text:
                    bot_response = analyze_text_content(extracted_text, file_name)
                else:
                    bot_response = "Could not read text from PDF."
            
            elif 'wordprocessingml' in file_type: # DOCX
                extracted_text = read_docx_text(file_content_bytes)
                if extracted_text:
                    bot_response = analyze_text_content(extracted_text, file_name)
                else:
                    bot_response = "Could not read text from DOCX."
            
            elif file_type == 'text/plain':
                decoded_text = file_content_bytes.decode('utf-8', errors='ignore')
                bot_response = analyze_text_content(decoded_text, file_name)
            
            else:
                bot_response = "Unsupported file type."

        elif user_message:
            full_prompt = f"""Act as a medical assistant. The user says: "{user_message}". 
            Provide:
            1. Symptoms Analysis
            2. Possible Causes (Disclaimer: Not a diagnosis)
            3. Home Remedies & Diet
            4. When to see a doctor
            """
            bot_response = call_gemini_api(full_prompt)

        return jsonify({'response': bot_response})

    except Exception as e:
        print(f"Server Error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)