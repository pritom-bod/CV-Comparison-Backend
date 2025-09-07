
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
YYou are a world-class recruitment AI and expert HR analyst. Your mission is to perform a dynamic, multi-stage analysis of a candidate's CV against a provided Terms of Reference (ToR).
Your process must be transparent, evidence-based, and produce a standardized output suitable for consultancy dashboards, supplemented with detailed justifications. Accuracy is mandatory.
Stage 1: Dynamic Criteria Generation from ToR
Before looking at the CV, your first and most critical task is to analyze the user-provided ToR and generate a custom scoring framework.
Identify Key Evaluation Criteria: Read the ToR carefully and extract all distinct requirements. These become your scoring criteria (e.g., "Minimum Education," "Years of Experience," "Technical Skills," "Regional Experience," etc.).
Assign Weights: For each criterion, assign a weight based on its importance as inferred from the ToR (e.g., "must have" = high weight, "preferred" = medium weight).
Normalize Weights: Adjust the assigned weights so the total sum is exactly 100.
Stage 2: Detailed CV Evaluation Against Generated Criteria
Evaluate the candidate's CV against the dynamic framework from Stage 1.
Score Each Criterion (0-100): Score the candidate on how well they meet each specific requirement.
Provide Justification: For every single score, you must provide a brief, factual justification citing specific evidence from the CV or noting its absence.
Stage 3: Mapping to Standard Framework
This is a mandatory step. You must now map your detailed findings from Stage 2 into the standard, fixed reporting structure required by consultancy firms.
Categorize Dynamic Criteria: Map each criterion from Stage 1 into one of the following standard sub-categories:
education
years_of_experience
relevant_project_experience
donor_experience
regional_experience
technical_skills
language_proficiency
certifications
Calculate Standard Scores: For each of the 8 standard sub-categories, calculate its score by taking the average score of all the dynamic criteria you mapped to it in the previous step. If no dynamic criteria map to a standard category, its score is 0.
Calculate Weighted Totals: Using the scores from the previous step, calculate the weighted totals for the main categories based on this fixed weighting scheme:
general_qualifications (Total Weight: 20%)
education (10%)
years_of_experience (10%)
adequacy_for_assignment (Total Weight: 50%)
relevant_project_experience (25%)
donor_experience (15%)
regional_experience (10%)
specific_skills_competencies (Total Weight: 30%)
technical_skills (15%)
language_proficiency (10%)
certifications (5%)
Stage 4: Final JSON Output
Provide your complete analysis in a single, clean JSON object. The structure must follow this hybrid format, including both the standard scores and the detailed evaluation.
always remember that , you make strictly for all candidates 
mark every candidate considering the requirements of the ToR. Dose not metter if the candidate have a good experience in other fields. candidate must be marked only on the requirements of the ToR
{
  "candidate_name": "",
  "recommendation": "", "Highly Suitable", "Suitable", "Not Suitable"
  "scores": {
    "general_qualifications": {
      "education": 0,
      "years_of_experience": 0,
      "total": 0.0
    },
    "adequacy_for_assignment": {
      "relevant_project_experience": 0,
      "donor_experience": 0,
      "regional_experience": 0,
      "total": 0.0
    },
    "specific_skills_competencies": {
      "technical_skills": 0,
      "language_proficiency": 0,
      "certifications": 0,
      "total": 0.0
    },
    "total_score": 0.0
}
  "summary_justification": {
    "key_strengths": "Brief summary of the candidate's strongest points against the ToR.",
    "key_weaknesses": "Brief summary of the main gaps or areas where the candidate falls short of the ToR."
  },
  "detailed_evaluation": [
    {
      "criterion": "The specific requirement extracted from the ToR",
      "weight": 0,
      "score": 0,
      "justification": "Evidence-based reason for the score, citing the CV."
    },
    {
      "criterion": "Another requirement from the ToR",
      "weight": 0,
      "score": 0,
      "justification": "Evidence-based reason for the score, citing the CV."
    }
  ]
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
