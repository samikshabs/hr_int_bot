import fitz  # PyMuPDF for PDF handling 
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize ChromaDB Client
chroma_client = chromadb.Client()
collection_name = "company_data"
collection = chroma_client.get_or_create_collection(collection_name)

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.

    Args:
    - pdf_file: File-like object representing the uploaded PDF.

    Returns:
    - str: Extracted text from the PDF.
    """
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def format_resume_text(resume_text):
    """
    Formats resume text to make main headings clear and visually appealing.
    
    Args:
    - resume_text (str): Raw text extracted from the resume.

    Returns:
    - str: Formatted text for better readability.
    """
    lines = resume_text.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line.isupper() and len(line.split()) < 6:
            formatted_lines.append(f"\n\n### {line} ###\n")
        elif line:  
            formatted_lines.append(line)

    return "\n".join(formatted_lines)

def extract_skills_using_ai(resume_text):
    """
    Extracts skills from the resume using the GPT-Neo model.

    Args:
    - resume_text (str): Full text of the resume.

    Returns:
    - str: Extracted skills as a string.
    """
    prompt = f"""
    You are a highly intelligent resume parser. Your task is to extract the skills section from the following resume text. 
    Return only the text below the 'Skills' section. If the 'Skills' section is not found, return an empty string.

    Resume Text:
    {resume_text}
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=600)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def process_resume_and_match_jobs(pdf_file):
    """
    Processes the resume and matches it with job descriptions in ChromaDB.

    Args:
    - pdf_file: File-like object representing the uploaded PDF.

    Returns:
    - dict: Dictionary containing matched jobs.
    """
    # Step 1: Extract raw text from the resume
    resume_text = extract_text_from_pdf(pdf_file)

    # Step 2: Retrieve job descriptions from ChromaDB
    results = collection.get(include=["documents", "metadatas"])
    job_descriptions = results["documents"]
    job_titles = [metadata["jobTitle"] for metadata in results["metadatas"]]

    if not job_descriptions:
        return {
            "matched_jobs": ["No job descriptions available in ChromaDB."]
        }

    # Step 3: Match the resume text with job descriptions using TF-IDF and cosine similarity
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        all_documents = [resume_text] + job_descriptions
        tfidf_matrix = vectorizer.fit_transform(all_documents)

        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        top_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 matches

        matched_jobs = [
            {"job_title": job_titles[i], "similarity": cosine_similarities[i]}
            for i in top_indices
        ]
    except ValueError as e:
        matched_jobs = [f"Error in processing job matching: {e}"]

    return {"matched_jobs": matched_jobs}
