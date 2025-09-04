
import os
import json
import google.generativeai as genai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document

load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

CV_COMPARISON_PROMPT = """
You are a world-class recruitment AI and expert HR analyst. 
Your task is to analyze a candidate's CV against a provided Terms of Reference (ToR) and produce a fully verified, structured evaluation. 
Accuracy is **mandatory**; there is **no room for errors, assumptions, or guesses**.  

---

### 1. **Check Input**
- If the CV is provided but the ToR is missing, respond exactly as:
  "I cannot complete your request. I have received the CV for [Candidate Name] but am missing the Terms of Reference (ToR) for the job assignment. Please provide the ToR so I can analyze the candidate's qualifications and score them against the requirements."
- Do **not** attempt to score or analyze without a valid ToR.

---

### 2. **Extract Candidate Information**
From the CV, extract **all explicit information only**:
- Full Name
- Education (degree, major, institution, graduation year)
- Total Years of Experience
- Relevant Project Experience (with details)
- Donor Experience (e.g., WB, ADB, UN)
- Regional Experience (countries/regions worked)
- Technical Skills
- Language Proficiency
- Certifications

⚠️ Only include data that is clearly present in the CV. **Never infer or assume anything.**

---

### 3. **Triple Verification**
Before producing the final output, perform the following **three-step verification**:
1. **Recheck the CV**: Ensure every extracted item is truly present.  
2. **Recheck against the ToR**: Ensure all scores and analysis strictly match the ToR requirements.  
3. **Recalculate totals**: Verify weighted scoring matches the ToR weighting exactly.  

---

### 4. **Scoring Framework**
Score according to this ToR structure:

- General Qualifications (20%)
  - Education (10%)
  - Years of Experience (10%)  
- Adequacy for Assignment (50%)
  - Relevant Project Experience (25%)
  - Donor Experience (15%)
  - Regional Experience (10%)  
- Specific Skills & Competencies (30%)
  - Technical Skills (15%)
  - Language Proficiency (10%)
  - Certifications (5%)  
- Total Score = 100%

---

### 5. **Strict Scoring Instructions**
- Each subcategory must be scored **0–100**.  
- Weighted totals must be calculated **step by step**.  
- Always confirm:
  - Each category subtotal = sum of its subcategories.  
  - Final total_score = sum of all category totals.  
  - Final total_score must be **exactly 100% maximum**.  
- If there is any mismatch, **adjust proportionally** and repeat the check until all totals are 100% accurate.

---

### 6. **JSON Output**
Provide output in this exact JSON format for dashboard integration:

```json
{
  "candidate_name": "",
  "scores": {
    "general_qualifications": {
      "education": 0,
      "years_of_experience": 0,
      "total": 0
    },
    "adequacy_for_assignment": {
      "relevant_project_experience": 0,
      "donor_experience": 0,
      "regional_experience": 0,
      "total": 0
    },
    "specific_skills_competencies": {
      "technical_skills": 0,
      "language_proficiency": 0,
      "certifications": 0,
      "total": 0
    },
    "total_score": 0
  },
  "recommendation": "" // e.g., "Highly Suitable", "Suitable", "Not Suitable"
}

```
"""

@csrf_exempt
@require_POST
def analyze_cv(request):
    """
    Handle CV analysis by extracting text from uploaded CV, combining with ToR,
    sending to Gemini API, and returning JSON response.
    """
    tor = request.POST.get('tor')
    cv_file = request.FILES.get('cv')

    # Validate inputs
    if not tor or not cv_file:
        return JsonResponse({'error': 'ToR and CV file are required'}, status=400)

    # Extract CV text
    try:
        cv_text = extract_cv_text(cv_file)
    except Exception as e:
        return JsonResponse({'error': f'Failed to extract CV text: {str(e)}'}, status=400)

    # Build full prompt
    full_prompt = CV_COMPARISON_PROMPT + f"\n\nInputs you will get:\n* Candidate CV Text: {cv_text}\n* ToR: {tor}\n\nEnsure the response is in valid JSON format wrapped in ```json ... ```."

    # Call Gemini API
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')  # Use text-focused model
        response = model.generate_content(full_prompt)

        ai_response = response.text

        # Check for missing ToR error
        if ai_response.startswith('I cannot complete your request'):
            return JsonResponse({'error': ai_response}, status=400)

        # Parse AI response as JSON
        try:
            # Strip markdown code blocks
            cleaned_response = ai_response.strip().strip('```json').strip('```').strip()
            result = json.loads(cleaned_response)
            required_keys = ['candidate_name', 'scores', 'recommendation']
            if not all(key in result for key in required_keys):
                raise ValueError('Invalid AI response format')
            return JsonResponse(result)
        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'AI response is not valid JSON: {str(e)}'}, status=500)
    except Exception as e:
        return JsonResponse({'error': f'Gemini API error: {str(e)}'}, status=500)

def extract_cv_text(cv_file):
    """
    Extract text from uploaded CV file (PDF, DOC, DOCX, or TXT).
    """
    file_ext = cv_file.name.split('.')[-1].lower()
    if file_ext == 'pdf':
        reader = PdfReader(cv_file)
        text = ''.join(page.extract_text() or '' for page in reader.pages)
    elif file_ext in ['doc', 'docx']:
        doc = Document(cv_file)
        text = '\n'.join(para.text for para in doc.paragraphs)
    elif file_ext == 'txt':
        text = cv_file.read().decode('utf-8')
    else:
        raise ValueError('Unsupported file type')
    return text.strip()
