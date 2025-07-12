# app.py
from flask import Flask, render_template, request, jsonify
import json
import requests
import base64
import io # Import io for BytesIO
import traceback # Import traceback for detailed error info
import os

# For PDF parsing
from pypdf import PdfReader

# For DOCX parsing
from docx import Document

app = Flask(__name__)

# --- Configuration ---
# For Running this code run with .env and storing the api : GEMINI_API_KEY=" "
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') #or paste your gemini 2.5 flash here in " "

# Check if the API key is set
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set the GEMINI_API_KEY environment variable before running the application.")

# --- Helper Functions for File Parsing ---
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
        traceback.print_exc() # Print full traceback
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
        traceback.print_exc() # Print full traceback
        return None

# --- Gemini API Call Functions ---
def call_gemini_api(prompt, model="gemini-2.5-flash"): # Using gemini-pro for text too, as it's multimodal
    """Calls the Gemini API for text generation."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    print(f"Calling Gemini API with model: {model}, prompt length: {len(prompt)}")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for bad status codes
        result = response.json()
        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            print("Gemini API call successful.")
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Unexpected Gemini API response structure: {json.dumps(result, indent=2)}")
            return "Sorry, I received an unexpected response from the AI. Please try again."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        traceback.print_exc() # Print full traceback
        return "Sorry, I couldn't connect to the AI service. Please try again later."
    except Exception as e:
        print(f"An unexpected error occurred during Gemini API call: {e}")
        traceback.print_exc() # Print full traceback
        return "An internal error occurred. Please try again."

def analyze_image_with_gemini(base64_image, mime_type):
    """Analyzes an image using Gemini Pro Vision."""
    # Using 'gemini-pro' for image analysis as well, as it supports multimodal input
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [
                {"text": "Analyze this medical image and provide:"},
                {"text": "1. **Description**: What the image shows"},
                {"text": "2. **Findings**: Notable observations"},
                {"text": "3. **Concerns**: Any potential issues"},
                {"text": "4. **Recommendations**: Suggested next steps"},
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": base64_image
                    }
                }
            ]
        }]
    }
    print(f"Calling Gemini Vision API for image (type: {mime_type}, size: {len(base64_image)} bytes).")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            print("Gemini Vision API call successful.")
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Unexpected Gemini Vision API response structure: {json.dumps(result, indent=2)}")
            return "Sorry, I received an unexpected response from the AI Vision service. Please try again."
    except requests.exceptions.RequestException as e:
        print(f"Error analyzing image with Gemini Vision API: {e}")
        traceback.print_exc() # Print full traceback
        return "Sorry, I couldn't process the image with the AI service. Please try again later."
    except Exception as e:
        print(f"An unexpected error occurred during image analysis: {e}")
        traceback.print_exc() # Print full traceback
        return "An internal error occurred during image analysis. Please try again."

def analyze_text_content(text_content, file_name=""):
    """Analyzes general text content (from files or user input)."""
    prompt = f"""Analyze the following medical report/text content from '{file_name}' and provide:
1. **Summary**: Brief overview of the findings
2. **Key Metrics**: Important values and their significance
3. **Potential Concerns**: Any abnormal values or findings
4. **Recommendations**: Suggested next steps or actions

Report content: {text_content}"""
    print(f"Analyzing text content from file '{file_name}' (length: {len(text_content)} characters).")
    return call_gemini_api(prompt)

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/chat')
def chat_page():
    """Renders the main chatbot HTML page."""
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat_api(): # Renamed to avoid conflict with route name
    """Handles chat messages and file uploads."""
    data = request.get_json()
    user_message = data.get('message')
    file_data_b64 = data.get('fileData') # Base64 encoded file content
    file_type = data.get('fileType')
    file_name = data.get('fileName')
    
    bot_response = ""
    print(f"Received chat request. Message: {user_message is not None}, File: {file_data_b64 is not None}")
    try:
        if file_data_b64 and file_type:
            print(f"Processing uploaded file: {file_name} (Type: {file_type})")
            file_content_bytes = base64.b64decode(file_data_b64)
            if file_type.startswith('image/'):
                bot_response = analyze_image_with_gemini(file_data_b64, file_type)
            elif file_type == 'application/pdf':
                extracted_text = read_pdf_text(file_content_bytes)
                if extracted_text:
                    bot_response = analyze_text_content(extracted_text, file_name)
                else:
                    bot_response = f"Could not extract text from PDF '{file_name}'. Please ensure it's a readable PDF or describe its content in text."
            elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': # .docx
                extracted_text = read_docx_text(file_content_bytes)
                if extracted_text:
                    bot_response = analyze_text_content(extracted_text, file_name)
                else:
                    bot_response = f"Could not extract text from DOCX '{file_name}'. Please ensure it's a valid DOCX file or describe its content in text."
            elif file_type == 'text/plain':
                # Decode as UTF-8 string for plain text
                decoded_text = file_content_bytes.decode('utf-8', errors='ignore')
                bot_response = analyze_text_content(decoded_text, file_name)
            else:
                bot_response = f"Unsupported file type for analysis: {file_type}. Please upload a PDF, DOCX, image, or plain text file."
        elif user_message:
            print(f"Processing text message: '{user_message[:50]}...'")
            full_prompt = f"""The user is describing a health issue. Provide comprehensive information based on their description, covering the following aspects clearly and concisely, using markdown for readability:
1.  **Symptoms:** List the symptoms associated with the described issue.
2.  **Possible Diseases/Conditions:** Suggest potential diseases or conditions that match the symptoms.
3.  **Home Remedies:** Suggest what action or food item can be taken at home to cure or feel better according to the disease.
4.  **Dietary Advice:** Recommend what can be eaten or avoided to help manage or cure the condition.
5.  **Medicines (General Advice):** Provide general types of over-the-counter or common medicines that might be used (stressing this is not medical advice and a doctor should be consulted).
6.  **Exercises/Activities:** Suggest exercises or activities that could be beneficial, or those to avoid.
7.  7. **Other Relevant Information:** Include any other important tips, precautions, or when to seek professional medical help.

User's health issue: "{user_message}"
"""
            bot_response = call_gemini_api(full_prompt)

        return jsonify({'response': bot_response})
    except Exception as e:
        print(f"Critical error in chat processing route: {e}")
        traceback.print_exc() # Print full traceback
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)

