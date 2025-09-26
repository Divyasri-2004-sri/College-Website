from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests, re, faiss, numpy as np, pytesseract
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from PIL import Image
from io import BytesIO
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

API_KEY = "AIzaSyAXsMPJ5lvdQiYa6_dTivE_KgbUn-02TJM"
genai.configure(api_key=API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your URLs
COLLEGE_URLS = [
    "https://www.idealtech.edu.in/website/Aboutus-vision_mission.php",
    "https://www.idealtech.edu.in/website/Aboutus-Our-Team.php",
    "https://www.idealtech.edu.in/website/Academics-Courses-Offered.php",
    "https://www.idealtech.edu.in/website/Academics-Calendar.php",
    "https://www.idealtech.edu.in/website/Academics-Syllabus.php",
    "https://www.idealtech.edu.in/website/Academics-Laboratories.php",
    "https://www.idealtech.edu.in/website/Academics-Scholarship.php",
    "https://www.idealtech.edu.in/website/Academics-Examcell.php",
    "https://www.idealtech.edu.in/website/Academics-Antiragging.php",
    "https://www.idealtech.edu.in/website/Academics-Code-of-Conduct.php",
    "https://www.idealtech.edu.in/website/Academics-PO's.php",
    "https://www.idealtech.edu.in/website/Academics-PSO's.php",
    "https://www.idealtech.edu.in/website/Academics-Alumni.php",
    "https://www.idealtech.edu.in/website/Departments-CSE.php",
    "https://www.idealtech.edu.in/website/Departments-AI.php",
    "https://www.idealtech.edu.in/website/Departments-Me.php",
    "https://www.idealtech.edu.in/website/Departments-ECE.php",
    "https://www.idealtech.edu.in/website/Departments-Civil.php",
    "https://www.idealtech.edu.in/website/Departments-EEE.php",
    "https://www.idealtech.edu.in/website/Departments-Humanities-&-Basic-Sciences.php",
    "https://www.idealtech.edu.in/website/Placements-home.php",
    "https://www.idealtech.edu.in/website/Placements-about.php",
    "https://www.idealtech.edu.in/website/Placements-Ourplacements.php",
    "https://www.idealtech.edu.in/website/Campuslife-NSS.php",
    "https://www.idealtech.edu.in/website/Campuslife-Library.php",
    "https://www.idealtech.edu.in/website/Campuslife-Transport.php",
    "https://www.idealtech.edu.in/website/Campuslife-Parking.php",
    "https://www.idealtech.edu.in/website/Campuslife-Canteen.php",
    "https://www.idealtech.edu.in/website/Campuslife-Sports.php",
    "https://www.idealtech.edu.in/website/Campuslife-Events.php",
    "https://www.idealtech.edu.in/website/Campuslife-Playground.php",
    "https://www.idealtech.edu.in/website/NAAC.php",
    "https://www.idealtech.edu.in/website/Stakeholders.php",
    "https://www.idealtech.edu.in/website/Committees.php"
]

# FAISS & embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
texts_store = []
roles_store = {}

# Text extraction
def extract_text_from_url(url):
    try:
        res = requests.get(url, timeout=15)
        if url.endswith(".pdf"):
            pdf = PdfReader(BytesIO(res.content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
        else:
            soup = BeautifulSoup(res.text, "html.parser")
            for s in soup(["script", "style"]): s.extract()
            for img_tag in soup.find_all("img"):
                img_url = img_tag.get("src")
                if img_url:
                    try:
                        img_resp = requests.get(img_url)
                        img = Image.open(BytesIO(img_resp.content))
                        text += " " + pytesseract.image_to_string(img)
                    except: pass
            return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return ""

def split_text(text, max_len=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, chunk = [], ""
    for s in sentences:
        if len(chunk)+len(s) > max_len:
            if chunk: chunks.append(chunk.strip())
            chunk = s
        else: chunk += " " + s
    if chunk: chunks.append(chunk.strip())
    return chunks

def add_to_vector_store(texts):
    global texts_store
    all_chunks = []
    for t in texts:
        all_chunks.extend(split_text(t))
    if not all_chunks: return
    embeddings = embedding_model.encode(all_chunks, convert_to_numpy=True)
    index.add(embeddings)
    texts_store.extend(all_chunks)

def retrieve(query, top_k=5):
    if len(texts_store) == 0: return []
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    results = [texts_store[i] for i in I[0] if i < len(texts_store)]
    return results

# Roles
def extract_roles(text):
    global roles_store
    for match in re.finditer(r"((?:Dr\.|Mr\.|Ms\.|Mrs\.)\s[\w\.]+\s?[\w\.]*)\s+(Principal|Director|Chairman)", text, re.I):
        roles_store[match.group(2).lower()] = match.group(1).strip()
    for match in re.finditer(r"((?:Dr\.|Mr\.|Ms\.|Mrs\.)\s[\w\.]+\s?[\w\.]*)\s+(?:HOD|Head of Department)", text, re.I):
        name = match.group(1).strip()
        dept_match = re.search(r"(CSE|Computer Science|EEE|ECE|Civil|Mechanical|AI|CS|Humanities)", text[match.end():match.end()+100], re.I)
        dept = dept_match.group(1).lower() if dept_match else "unknown"
        roles_store[f"{dept} hod"] = name

# Gemini answer
def generate_answer_gemini(prompt):
    try:
        # Updated model to a supported version
        model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        return response.text if response else "No response from Gemini."
    except Exception as e:
        return f"Error: {e}"

def answer_query(query):
    query_lower = query.lower()
    for role_key in roles_store:
        if role_key in query_lower:
            return roles_store[role_key]
    relevant_chunks = retrieve(query)
    context = " ".join(relevant_chunks) if relevant_chunks else ""
    prompt = f"Answer the user query using the context below:\nContext: {context}\nQuery: {query}\nAnswer clearly and concisely:"
    return generate_answer_gemini(prompt)

# Initialize
print("Fetching and indexing pages...")
for url in COLLEGE_URLS:
    text = extract_text_from_url(url)
    if text and len(text)>50:
        extract_roles(text)
        add_to_vector_store([text])
        print(f"Indexed: {url}")
print("Indexing complete.")

# Chat endpoint
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query", "")
    answer = answer_query(query)
    return {"answer": answer}
