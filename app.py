import PyPDF2
import re 
import requests 
from fastapi import FastAPI , UploadFile , File, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

#---------- PDF Extraction --------------
def extract_text_from_pdf(file_path):
    text=""
    with open(file_path,"rb") as file :
        reader = PyPDF2.PdfReader(file)
        
        for page in reader.pages:
            text+=page.extract_text()
            
    return text

#------------ Cleaning Text -----------------------
def clean_text(text):
    text=text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\-]','',text)
    text=re.sub(r'\s+',' ',text)
    
    return text

#----------- Job Skill Extraction using AI -------------
def extract_job_skills(job_desc):
    url="http://localhost:11434/api/generate"
    prompt= f"""
    Extract ONLY technical skills from this job description.
    Return ONLY a comma-separated list.
    No sentences. No explanation.
    {job_desc}
    """
    data = {
        'model':'llama3',
        'prompt':prompt,
        'stream':False
    }
    response = requests.post(url,json=data)
    skills_text=response.json()['response']
    skills_text=skills_text.replace("\n","").lower()
    skills=[s.strip().lower() for s in skills_text.split(',') if len(s.strip())>0]
    return skills

#--------------- Skill Extraction ---------------------------
def extract_skills(text):
    skills_map = {
        "python": ["python"],
        "java": ["java"],
        "c++": ["c++"],
        "sql": ["sql"],
        "machine learning": ["machine learning", "ml"],
        "deep learning": ["deep learning", "dl"],
        "data analysis": ["data analysis", "data analytics"],
        "nlp": ["nlp", "natural language processing"],
        "tensorflow": ["tensorflow"],
        "keras": ["keras"],
        "pandas": ["pandas"],
        "scikit-learn": ["scikit-learn", "sklearn"]
    }

    found_skills = set()

    for main_skill, variations in skills_map.items():
        for v in variations:
            if v in text:
                found_skills.add(main_skill)

    return list(found_skills)

#--------------- Matching Skills -----------------------
def match_skills(resume_skills , job_skills):
    matched=[]
    missing=[]
    resume_set=set([s.lower() for s in resume_skills])
    for skill in job_skills:
        skill_clean=skill.lower()
        if any(skill_clean in r or r in skill_clean for r in resume_set):
            matched.append(skill)
        else:
            missing.append(skill)
            
    return matched,missing

#--------------------- LLM Feedback (Ollama)--------------
def get_ai_feedback(text):
    url = "http://localhost:11434/api/generate"
    prompt = f"""
    You are an AI resume analyzer.
    Analyze the following resume and give feedback :
    {text}
    Give structured output :
    1. Strengths:
         - ......
    2. Weaknesses:
        - .....
    3. Suggestions:
        - .....
    4. Missing Skills for job role:
        - ......
    
    Be concise and professional.
    """
    data = {
        "model":"llama3",
        "prompt" : prompt ,
        "stream" : False 
    } 
    
    response = requests.post(url,json=data)
    return response.json()["response"]

#--------------- API ------------------
@app.post("/analyze")
async def analyze_resume(file:UploadFile=File(...),
                         job_description:str=Form(...)):
    pdf=PyPDF2.PdfReader(file.file)
    text=""
    for page in pdf.pages :
        text+=page.extract_text()
        
    cleaned = clean_text(text)
    skills = extract_skills(cleaned)
    
    job_skills = extract_job_skills(job_description)
    matched , missing = match_skills(skills , job_skills)
    
    feedback = get_ai_feedback(cleaned)
    feedback = feedback.replace("**","")
    return {
        'skills':skills,
        'matched':matched,
        'missing':missing,
        'ai_feedback':feedback
    }
    