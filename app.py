from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import time
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ----------------------------
# Environment & App
# ----------------------------
app = FastAPI(title="UM6P Admissions RAG API")

# ----------------------------
# Load Gemini
# ----------------------------
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

# ----------------------------
# Load Retriever (ONCE)
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("um6p_faiss_index.bin")

with open("um6p_chunk_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# ----------------------------
# Request / Response Models
# ----------------------------
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list
    retrieval_time: float
    generation_time: float
    total_time: float

# ----------------------------
# Core RAG Logic (UNCHANGED)
# ----------------------------
def retrieve(question, k=7):
    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue

        chunk = metadata[idx]
        similarity = 1 / (1 + dist)

        results.append({
            "content": chunk["content"],
            "source_type": chunk["source_type"],
            "similarity": similarity
        })

    return results

def generate_answer(context, question):
    prompt = f"""
You are an expert UM6P admissions assistant.

Context:
{context}

Question: {question}

Answer:
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=1500
        )
    )

    return response.text.strip()

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_question(req: QuestionRequest):
    start = time.time()

    # Retrieve
    r_start = time.time()
    results = retrieve(req.question)
    retrieval_time = time.time() - r_start

    context = "\n---\n".join(
        f"[{i+1}] {r['content']}" for i, r in enumerate(results)
    )

    # Generate
    g_start = time.time()
    answer = generate_answer(context, req.question)
    generation_time = time.time() - g_start

    return {
        "answer": answer,
        "sources": results[:3],
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": time.time() - start
    }

@app.get("/")
def health_check():
    return {"status": "UM6P RAG API is running"}
