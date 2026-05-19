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
load_dotenv(override=True)

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

def call_gemini_api(prompt, model_index=0):
    """Calls the Gemini API for text generation with fallback."""
    models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3-flash-preview"]
    if model_index >= len(models):
        return "System error: All AI models failed to respond."
        
    model = models[model_index]
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
            print(f"API Error {response.status_code} for {model}: {response.text}")
            print("Attempting fallback to next model...")
            return call_gemini_api(prompt, model_index + 1)

        result = response.json()
        if result.get('candidates') and result['candidates'][0].get('content'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Empty response from {model}, trying next model...")
            return call_gemini_api(prompt, model_index + 1)
            
    except Exception as e:
        print(f"Gemini API Exception for {model}: {e}")
        traceback.print_exc()
        return call_gemini_api(prompt, model_index + 1)

def analyze_image_with_gemini(base64_image, mime_type, profile_context="", model_index=0):
    """Analyzes an image using Gemini Vision with fallback."""
    models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3-flash-preview"]
    if model_index >= len(models):
        return "System error: All AI vision models failed to respond."
        
    model = models[model_index]
    if not GEMINI_API_KEY:
        return "Error: API Key is missing on server."

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""Analyze this medical image. {profile_context}Provide:
1. **Description**: Describe what the image shows.
2. **Findings**: Detail the key observations or indicators, highlighting abnormal findings if visible.
3. **Concerns**: Mention any worrying or abnormal markers.
4. **Recommendations**: Advise on next steps.

Remember: Provide educational/informational analysis only. This is not a formal diagnosis.
"""
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": base64_image
                    }
                }
            ]
        }]
    }
    
    print(f"Calling Gemini Vision API with model: {model}...")
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"Vision API Error {response.status_code} for {model}: {response.text}")
            return analyze_image_with_gemini(base64_image, mime_type, profile_context, model_index + 1)

        result = response.json()
        if result.get('candidates') and result['candidates'][0].get('content'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Empty vision response from {model}, trying next model...")
            return analyze_image_with_gemini(base64_image, mime_type, profile_context, model_index + 1)
            
    except Exception as e:
        print(f"Vision Exception for {model}: {e}")
        return analyze_image_with_gemini(base64_image, mime_type, profile_context, model_index + 1)


def analyze_text_content(text_content, file_name="", profile_context=""):
    """Analyzes general text content."""
    prompt = f"""Analyze the following medical report/text from '{file_name}':
    {profile_context}
    
    Provide:
    1. **Summary**: A high-level overview of the report.
    2. **Key Metrics**: A bulleted list of key biomarkers or parameters, highlighting any abnormal values (high/low).
    3. **Potential Concerns**: What should the patient pay attention to?
    4. **Recommendations**: Recommended next steps or follow-ups.

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
    user_profile = data.get('profile', {})
    
    # Build profile context to personalize the chatbot's response
    profile_context = ""
    if user_profile:
        profile_parts = []
        if user_profile.get('age'):
            profile_parts.append(f"Age: {user_profile.get('age')}")
        if user_profile.get('gender'):
            profile_parts.append(f"Gender: {user_profile.get('gender')}")
        if user_profile.get('allergies') and user_profile.get('allergies').strip().lower() != 'none':
            profile_parts.append(f"Known Allergies: {user_profile.get('allergies')}")
        if user_profile.get('conditions') and user_profile.get('conditions').strip().lower() != 'none':
            profile_parts.append(f"Pre-existing conditions: {user_profile.get('conditions')}")
        
        if profile_parts:
            profile_context = "Patient Background Context:\n" + "\n".join(f"- {part}" for part in profile_parts) + "\n\n"
    
    bot_response = ""
    
    try:
        if file_data_b64 and file_type:
            print(f"Processing file: {file_name}")
            file_content_bytes = base64.b64decode(file_data_b64)
            
            if file_type.startswith('image/'):
                bot_response = analyze_image_with_gemini(file_data_b64, file_type, profile_context)
            
            elif file_type == 'application/pdf':
                extracted_text = read_pdf_text(file_content_bytes)
                if extracted_text:
                    bot_response = analyze_text_content(extracted_text, file_name, profile_context)
                else:
                    bot_response = "Could not read text from PDF."
            
            elif 'wordprocessingml' in file_type: # DOCX
                extracted_text = read_docx_text(file_content_bytes)
                if extracted_text:
                    bot_response = analyze_text_content(extracted_text, file_name, profile_context)
                else:
                    bot_response = "Could not read text from DOCX."
            
            elif file_type == 'text/plain':
                decoded_text = file_content_bytes.decode('utf-8', errors='ignore')
                bot_response = analyze_text_content(decoded_text, file_name, profile_context)
            
            else:
                bot_response = "Unsupported file type."

        elif user_message:
            full_prompt = f"""{profile_context}Act as a medical assistant. The user says: "{user_message}". 
            Provide:
            1. Symptoms Analysis
            2. Possible Causes (Disclaimer: Not a diagnosis)
            3. Home Remedies & Diet
            4. When to see a doctor
            
            Keep in mind the Patient Background Context (if provided above) to customize your analysis, warnings, and remedy suggestions (especially checking for potential drug/herb-allergy interactions or exacerbation of chronic conditions).
            """
            bot_response = call_gemini_api(full_prompt)

        return jsonify({'response': bot_response})

    except Exception as e:
        print(f"Server Error: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analyze-skin', methods=['POST'])
def analyze_skin_api():
    """Dedicated endpoint for skin analysis using Gemini Vision."""
    data = request.get_json()
    image_data_b64 = data.get('imageData')
    mime_type = data.get('mimeType', 'image/jpeg')
    user_profile = data.get('profile', {})

    if not image_data_b64:
        return jsonify({'error': 'No image data provided'}), 400

    if not GEMINI_API_KEY:
        return jsonify({'error': 'API Key is missing on server.'}), 500

    # Build profile context
    profile_context = ""
    if user_profile:
        profile_parts = []
        if user_profile.get('age'):
            profile_parts.append(f"Age: {user_profile.get('age')}")
        if user_profile.get('gender'):
            profile_parts.append(f"Gender: {user_profile.get('gender')}")
        if user_profile.get('allergies') and user_profile.get('allergies').strip().lower() != 'none':
            profile_parts.append(f"Known Allergies: {user_profile.get('allergies')}")
        if user_profile.get('conditions') and user_profile.get('conditions').strip().lower() != 'none':
            profile_parts.append(f"Pre-existing Conditions: {user_profile.get('conditions')}")
        if profile_parts:
            profile_context = "Patient Background:\n" + "\n".join(f"- {p}" for p in profile_parts) + "\n\n"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}

    skin_prompt = f"""You are a knowledgeable dermatology assistant. Analyze the skin shown in this image.
{profile_context}
Please provide a structured response with the following sections:

## 🔍 Skin Observation
Describe what you can visually observe — skin tone, texture, visible marks, redness, dryness, lesions, rashes, spots, discoloration, or other notable features.

## 🩺 Possible Skin Conditions
List any potential skin conditions or issues that the observed features may suggest (e.g. acne, eczema, psoriasis, hyperpigmentation, rosacea, fungal infection, contact dermatitis, dry skin, sun damage, etc.). Be informative but non-alarmist.

## ⚠️ Warning Signs
Highlight any features that may warrant prompt medical attention (e.g. asymmetric moles, irregular borders, multiple colors in a lesion, rapidly spreading rash, open sores, etc.).

## 💡 Skincare Tips & Home Remedies
Provide 4–6 practical skincare tips, lifestyle adjustments, or gentle home remedies relevant to what you observe.

## 🏥 When to See a Dermatologist
Explain clearly when the person should seek professional dermatological evaluation.

**Important Disclaimer:** This analysis is for educational and informational purposes only. It is NOT a medical diagnosis. Always consult a licensed dermatologist or healthcare provider for proper evaluation and treatment.
"""

    payload = {
        "contents": [{
            "parts": [
                {"text": skin_prompt},
                {"inlineData": {"mimeType": mime_type, "data": image_data_b64}}
            ]
        }]
    }

    SKIN_MODELS = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-3-flash-preview"]

    print("Calling Gemini Vision API for skin analysis...")
    last_error = None
    for model_name in SKIN_MODELS:
        model_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
        try:
            response = requests.post(model_url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                if result.get('candidates') and result['candidates'][0].get('content'):
                    analysis = result['candidates'][0]['content']['parts'][0]['text']
                    return jsonify({'response': analysis})
                else:
                    return jsonify({'error': 'No content returned from AI.'}), 500
            else:
                last_error = f"Model {model_name} returned {response.status_code}"
                print(f"Skin Analysis: {last_error}, trying next model...")
                continue
        except Exception as e:
            last_error = str(e)
            print(f"Skin analysis exception with {model_name}: {e}")
            traceback.print_exc()
            continue

    return jsonify({'error': f'All models failed. Last error: {last_error}'}), 500


@app.route('/api/explain', methods=['POST'])
def explain_api():
    data = request.get_json()
    term = data.get('term')
    if not term:
        return jsonify({'error': 'No term provided'}), 400
    
    prompt = f"""Explain the medical term '{term}' in simple layman's language. 
    Provide:
    1. A simple definition (1-2 sentences).
    2. Why it matters or what it relates to (1 sentence).
    Keep the explanation extremely brief, clear, and easy for a non-medical person to understand. Avoid using other complex medical jargon in the explanation.
    """
    explanation = call_gemini_api(prompt)
    return jsonify({'explanation': explanation})

if __name__ == '__main__':
    app.run(debug=True)