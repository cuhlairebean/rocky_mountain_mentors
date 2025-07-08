# Gradio ALICIA app
# Claire H Levitt 7 8 2025 | Rocky Mountain Mentors


import os
import re
import numpy as np
import tiktoken
import gradio as gr
from dotenv import load_dotenv
from pathlib import Path
import openai
from sklearn.metrics.pairwise import cosine_similarity

# Load environment and OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths and constants
PROJECT_ROOT = Path.cwd()
CORPUS_PATH = PROJECT_ROOT / "data" / "rmm_corpus" / "resources.txt"
AGENT_DESC_PATH = PROJECT_ROOT / "data" / "rmm_corpus" / "agent_description.txt"

EMBED_MODEL = "text-embedding-3-small"
TOKENIZER = tiktoken.encoding_for_model("gpt-4o")
MAX_TOKENS_CONTEXT = 3000

# Load corpus
raw_text = CORPUS_PATH.read_text(encoding="utf-8")
passages = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

def embed(texts):
    resp = openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype="float32")

emb_vectors = embed(passages)

def retrieve(query, k=4):
    q_vec = embed([query])[0].reshape(1, -1)
    similarities = cosine_similarity(q_vec, emb_vectors)[0]
    idxs = similarities.argsort()[::-1][:k]
    return [(passages[i], float(similarities[i])) for i in idxs]

SYSTEM_DESC = AGENT_DESC_PATH.read_text(encoding="utf-8").strip()

conversation = []
student_profile = {}

def parse_student_info(text):
    prog = None
    year = None
    program_match = re.search(r"\b(PhD|PHD|MS|M\.?S\.?|Bachelor'?s?)\b", text, re.I)
    if program_match:
        prog = program_match.group(1).upper().replace(".", "")
        if prog.startswith("B"):
            prog = "Bachelor's"
    year_match = re.search(r"\b([1-6])(?:st|nd|rd|th)?\s*year\b", text, re.I)
    if year_match:
        year = int(year_match.group(1))
    else:
        words = {"freshman":1,"sophomore":2,"junior":3,"senior":4}
        for w,n in words.items():
            if w in text.lower():
                year = n
                break
    return prog, year

def build_prompt(user_message):
    sys_base = {"role": "system", "content": SYSTEM_DESC}
    if student_profile:
        sys_personal = {"role": "system",
                        "content": f"Student program: {student_profile['program']}, Year: {student_profile['year']}."}
    else:
        sys_personal = {"role": "system",
                        "content": "Ask once for program + year, then remember."}

    docs = retrieve(user_message, k=5)
    context = "\n\n".join(f"{i+1}. {d[0]}" for i, d in enumerate(docs))

    assistant_context = {
        "role": "assistant",
        "content": (
            "**Grounding data – you MUST base your answer ONLY on these excerpts. "
            "If they don’t contain the answer, reply 'I don’t have that information, but I searched online and found {then search online and find a reliable source like the program website, find the answer and return the answer AND your source link}. "
            "Everything should be specific to the University of Colorado Anschutz Medical Campus and Denver campus, as well as the student's current program and year.'**\n\n"
            + context)
    }

    return [sys_base, sys_personal] + conversation + [
        assistant_context,
        {"role": "user", "content": user_message}
    ]

def chatbot(message, history):
    # Initialize history if empty
    history = history or []

    if not message.strip():
        # Return empty input and existing history unchanged
        return "", history

    # Parse student info only once from history or message
    # We'll try to extract profile from history first
    student_profile = None
    for msg in history:
        prog, yr = parse_student_info(msg["content"])
        if prog and yr:
            student_profile = {"program": prog, "year": yr}
            break
    # If not found in history, try current message
    if not student_profile:
        prog, yr = parse_student_info(message)
        if prog and yr:
            student_profile = {"program": prog, "year": yr}

    # Build the prompt conversation based on history + current message
    # Convert gradio history (list of dicts) to OpenAI message format with roles
    conversation = [{"role": m["role"], "content": m["content"]} for m in history]

    sys_base = {"role": "system", "content": SYSTEM_DESC}
    if student_profile:
        sys_personal = {"role": "system",
                        "content": f"Student program: {student_profile['program']}, Year: {student_profile['year']}."}
    else:
        sys_personal = {"role": "system",
                        "content": "Ask once for program + year, then remember."}

    docs = retrieve(message, k=5)
    context = "\n\n".join(f"{i+1}. {d[0]}" for i, d in enumerate(docs))

    assistant_context = {
        "role": "assistant",
        "content": (
            "**Grounding data – you MUST base your answer ONLY on these excerpts. "
            "If they don’t contain the answer, reply 'I don’t have that information, but I searched online and found {then search online and find a reliable source like the program website, find the answer and return the answer AND your source link}. "
            "Everything should be specific to the University of Colorado Anschutz Medical Campus and Denver campus, as well as the student's current program and year.'**\n\n"
            + context)
    }

    # Compose the full prompt messages for OpenAI
    messages = [sys_base, sys_personal] + conversation + [assistant_context, {"role": "user", "content": message}]

    # Call OpenAI chat completion
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=512
    ).choices[0].message.content.strip()

    # Append user message and assistant response to history for Gradio
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})

    # Return empty input box and updated chat history
    return "", history


demo = gr.ChatInterface(
    fn=chatbot,
    title="ALICIA: Academic Learning and Institutional Coaching Intelligent Assistant",
    description="Ask about programs, mentorship, resources at CU Anschutz and Denver.",
    type="messages"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=True)
