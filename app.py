import os
import time
import faiss
import pickle
import gradio as gr
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import re

# ----------------------------
# Gemini Setup
# ----------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

# ----------------------------
# Load RAG Assets (ONCE)
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("um6p_faiss_index.bin")

with open("um6p_chunk_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# ----------------------------
# Utilities
# ----------------------------
def clean_markdown(text: str) -> str:
    """
    Cleans Gemini output for user-friendly Markdown rendering
    """
    text = text.strip()

    # Normalize bullets
    text = re.sub(r"\n\s*[-•]\s*", "\n- ", text)

    # Remove repeated empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def confidence_label(avg_similarity: float) -> str:
    if avg_similarity >= 0.6:
        return "🟢 **High confidence** (official & consistent sources)"
    elif avg_similarity >= 0.45:
        return "🟡 **Medium confidence** (mixed sources)"
    else:
        return "🔴 **Low confidence** (limited official information)"

# ----------------------------
# Retrieval
# ----------------------------
def retrieve(question, k=7):
    q_emb = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(q_emb, k)

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

# ----------------------------
# Generation
# ----------------------------
def generate_answer(context, question):
    prompt = f"""
You are a friendly and professional UM6P admissions assistant.

Answer clearly, step-by-step when appropriate.
Use bullet points and short sections.
Do NOT mention the word "context" in your answer.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=1200
        )
    )

    return clean_markdown(response.text)

# ----------------------------
# Main RAG Pipeline
# ----------------------------
def rag_pipeline(question):
    start = time.time()

    results = retrieve(question)
    avg_sim = sum(r["similarity"] for r in results) / max(len(results), 1)

    context = "\n---\n".join(
        f"[{i+1}] {r['content']}" for i, r in enumerate(results)
    )

    answer = generate_answer(context, question)

    sources_md = ""
    for i, r in enumerate(results[:3]):
        sources_md += (
            f"**Source {i+1} — {r['source_type']}**  \n"
            f"_Similarity_: `{r['similarity']:.3f}`  \n"
            f"{r['content'][:600]}...\n\n"
        )

    stats_md = (
        f"**Confidence:** {confidence_label(avg_sim)}  \n"
        f"⏱ **Total time:** {time.time() - start:.2f}s"
    )

    return answer, sources_md, stats_md

# ----------------------------
# UI (Gradio Blocks)
# ----------------------------
with gr.Blocks(css="""
body { font-family: Inter, sans-serif; }
.card { border-radius: 16px; padding: 20px; background: #0f172a; }
""") as demo:

    gr.Markdown("""
# 🎓 UM6P Admissions Assistant  
Ask questions about **admissions, documents, arrival, and procedures**.  
Answers are based on **verified sources**.
""")

    question = gr.Textbox(
        label="Ask a question",
        placeholder="What are the admission requirements for UM6P?",
        lines=2
    )

    ask_btn = gr.Button("Ask", variant="primary")

    with gr.Group():
        answer_md = gr.Markdown(label="Answer")
        confidence_md = gr.Markdown(label="Confidence")
        with gr.Accordion("🔍 View sources", open=False):
            sources_md = gr.Markdown()

    ask_btn.click(
        fn=rag_pipeline,
        inputs=question,
        outputs=[answer_md, sources_md, confidence_md]
    )

demo.launch()
