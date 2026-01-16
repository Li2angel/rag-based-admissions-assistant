import os
import time
import re
import gradio as gr
import google.generativeai as genai

# ----------------------------
# IMPORT YOUR RETRIEVER
# ----------------------------
from um6p_retriever import UM6PRetriever

# ----------------------------
# Gemini Setup
# ----------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("models/gemini-2.5-flash")

# ----------------------------
# Initialize Retriever (ONCE)
# ----------------------------
retriever = UM6PRetriever()

# ----------------------------
# Utility Functions
# ----------------------------
def clean_markdown(text: str) -> str:
    """Clean and normalize Gemini output for Markdown UI"""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n\s*[-•]\s*", "\n- ", text)
    return text
    

# ----------------------------
# LLM Generation
# ----------------------------
def generate_answer(context: str, question: str) -> str:
    prompt = f"""
You are an expert UM6P admissions assistant. Provide a COMPLETE and detailed answer using the context.

Instructions:
- Provide a COMPLETE answer using ALL available details from the context.
- Use bullet points for readability.
- Use the document names (e.g., "According to [Filename]...") when answering.
- If the context covers multiple aspects, include them all
- DO NOT stop mid-sentence - complete all points
- If the answer isn't in the context, say you don't know—do not hallucinate.
- Be concise but complete
- Do NOT mention internal context or sources explicitly

Information:
{context}

User question:
{question}

Answer:
"""

    response = llm.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=1500,
            stop_sequences=None
        )
    )

    return clean_markdown(response.text)


# ----------------------------
# Main RAG Pipeline
# ----------------------------
def rag_pipeline(question):
    start_time = time.time()

    # 1. Retrieve (Now uses k=7 as per your evaluation)
    results = retriever.search(question, k=7)

    if not results:
        return "⚠️ I couldn’t find relevant official information for this question.", "", "🔴 Low confidence"

    # 2. Confidence Logic (Adjusted for BGE-Small L2 distances)
    # Typically: < 0.4 is High, 0.4 - 0.7 is Medium, > 0.7 is Low
    
    best_dist = results[0]['similarity'] # Assuming this is the raw FAISS distance
    if best_dist < 0.4:
        conf_str = "🟢 **High confidence** — Found exact document matches"
    elif best_dist < 0.6:
        conf_str = "🟡 **Medium confidence** — Relevant but general info found"
    else:
        conf_str = "🔴 **Low confidence** — No direct document match"

    # 3. Build context with actual filenames
    context = retriever.get_context_for_llm(results)

    # 4. Generate answer
    answer = generate_answer(context, question)

    # 5. Sources UI - Showing FILENAMES instead of "Source 1"
    sources_md = "### 📚 References Found:\n"
    for r in results[:3]: # Show top 3 for brevity
        fname = r.get('filename', 'Official Document')
        sources_md += (
            f"📁 **{fname}**\n"
            f"> {r['content'][:300]}...\n\n"
        )

    stats_md = f"{conf_str}  \n⏱ **Process time:** {time.time() - start_time:.2f}s"

    return answer, sources_md, stats_md


# ----------------------------
# Sample Questions Data
# ----------------------------
example_map = {
    "Application Stages": "What are the different stages of the application process?",
    "Visa Documents": "What documents are required for a student visa?",
    "Scholarship Timeline": "When is the deadline for scholarship applications?",
    "Airport Arrival": "How can I arrange for an airport pickup upon arrival?",
    "Interview Prep": "How should I prepare for the admissions interview?",
    "Visa Fee": "What is the cost of the student visa application?"
}

# Modern Dashboard CSS with high-contrast text fixes
custom_css = """
/* Center the main header and logo */
.centered-header { 
    display: flex; 
    flex-direction: column; 
    align-items: center; 
    text-align: center; 
    margin-bottom: 30px;
    width: 100%;
}

/* Fix text visibility: Ensure body text inherits accessible theme colors */
.gradio-container {
    font-family: 'Inter', system-ui, sans-serif;
}

/* Dashboard Cards with subtle shadow and clear borders */
.modern-card { 
    border-radius: 16px !important; 
    border: 1px solid var(--border-color-primary) !important; 
    background-color: var(--background-fill-primary) !important; 
    padding: 24px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
}

/* Ensure Markdown text is always visible */
.prose {
    color: var(--body-text-color) !important;
}

/* High-visibility headers */
h1, h2, h3 {
    color: var(--body-text-color-subdued) !important;
    font-weight: 700 !important;
}

/* UM6P Orange Status Tag */
.status-pill {
    background: #fff7ed;
    color: #f58a1f; /* Official UM6P Orange */
    padding: 6px 16px;
    border-radius: 50px;
    font-size: 0.85em;
    font-weight: 700;
    border: 1px solid #fed7aa;
    display: inline-block;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="orange",     
        neutral_hue="slate",      
        spacing_size="md",
        radius_size="lg",
    ),
    css=custom_css, # Uses your provided custom_css
    title="Admissions Assistant"
) as demo:
    
    # 🏛️ Branded Header: Centered
    with gr.Group(elem_classes="centered-header"):
        gr.Image("logo.png", show_label=False, width=170, container=False)
        gr.Markdown("# Admissions Assistant")
        gr.Markdown("#### Your AI-powered guide for Mohammed VI Polytechnic University admissions")

    # 🖥️ Dashboard Grid
    with gr.Row(equal_height=True):
        
        # Left Panel: Query & Custom Examples
        with gr.Column(scale=2, elem_classes="modern-card"):
            gr.Markdown("### Submit your Enquiry")
            question_input = gr.Textbox(
                label=None, 
                placeholder="How do I prepare for the interview?", 
                lines=4,
                container=False
            )
            submit_btn = gr.Button("Find Verified Answer", variant="primary")
            
            gr.Markdown("### Quick Questions")
            
            # Create a Grid of Buttons with short names
            with gr.Row():
                btn1 = gr.Button("Application Stages", size="sm")
                btn2 = gr.Button("Visa Documents", size="sm")
                btn3 = gr.Button("Scholarship Timeline", size="sm")
            with gr.Row():
                btn4 = gr.Button("Airport Arrival", size="sm")
                btn5 = gr.Button("Interview Prep", size="sm")
                btn6 = gr.Button("Visa Fee", size="sm")
        
        # Right Panel: Results Area
        with gr.Column(scale=3, elem_classes="modern-card"):
            gr.Markdown("### Assistant Response")
            answer_output = gr.Markdown("_The assistant is ready for your inquiry..._")
            
            with gr.Row():
                gr.HTML("<div class='status-pill'>System Ready</div>")

    # 📚 Source Transparency
    with gr.Row():
        with gr.Column():
            with gr.Accordion("Verified Reference Documents", open=False):
                sources_output = gr.Markdown("Retrieved document citations will appear here.")

    # 🔗 Footer
    gr.Markdown("<center><small>© 2026 Mohammed VI Polytechnic University</small></center>")

    # ----------------------------
    # 3. Logic & Event Handling
    # ----------------------------
    
    # Helper to load the full question from the button label
    def load_full_question(label_text):
        return example_map.get(label_text, label_text)

    # Connect buttons: Update textbox -> then trigger pipeline
    example_btns = [btn1, btn2, btn3, btn4, btn5, btn6]
    for btn in example_btns:
        btn.click(
            fn=load_full_question, 
            inputs=[btn], 
            outputs=[question_input]
        ).then(
            fn=rag_pipeline, 
            inputs=[question_input], 
            outputs=[answer_output, sources_output]
        )

    # Standard submit logic
    submit_btn.click(fn=rag_pipeline, inputs=question_input, outputs=[answer_output, sources_output])
    question_input.submit(fn=rag_pipeline, inputs=question_input, outputs=[answer_output, sources_output])

if __name__ == "__main__":
    demo.launch()