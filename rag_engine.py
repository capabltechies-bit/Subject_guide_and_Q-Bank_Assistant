"""
rag_engine.py
RAG (Retrieval-Augmented Generation) engine for the Subject Guide & QBank Assistant.
Uses FAISS vector search + Google Gemini to answer academic questions.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv
from vector_store import search

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Use Gemini 1.5 Flash for fast, cost-effective generation
_MODEL_NAME = "gemini-2.5-flash"


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['source']} ({chunk['content_type']})]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _call_gemini(system_prompt: str, user_message: str) -> str:
    """Send a prompt to Gemini and return the text response."""
    model = genai.GenerativeModel(
        model_name=_MODEL_NAME,
        system_instruction=system_prompt,
    )
    response = model.generate_content(user_message)
    return response.text


# ── Topic Explanation ──────────────────────────────────────────────────────────

TOPIC_SYSTEM_PROMPT = """\
You are an expert academic tutor helping students understand complex topics clearly.

You will be given:
1. A QUESTION or topic the student wants to understand.
2. CONTEXT extracted from their uploaded study materials (textbooks, notes, slides).

Your task:
- Explain the topic in a clear, structured, and student-friendly way.
- Use the provided context as your primary source of information.
- Include relevant definitions, key concepts, examples, and diagrams (described in text if needed).
- If the context doesn't fully cover the topic, supplement with your own knowledge but note it.
- Use Markdown formatting: headings, bullet points, bold for key terms, code blocks where relevant.
- End with a brief "Key Takeaways" summary.

Do NOT just copy-paste from the context. Synthesize and explain it well.
"""


def answer_topic(query: str, k: int = 6) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks and generate a topic explanation.

    Returns:
        answer (str): Markdown-formatted explanation.
        sources (list): The retrieved chunk dicts used as context.
    """
    sources = search(query, k=k)

    if not sources:
        return (
            "⚠️ No relevant content found in your uploaded documents. "
            "Please make sure you've uploaded materials related to this topic.",
            [],
        )

    context = _build_context(sources)

    user_message = f"""QUESTION / TOPIC:
{query}

CONTEXT FROM STUDY MATERIALS:
{context}

Please explain this topic thoroughly based on the context above."""

    answer = _call_gemini(TOPIC_SYSTEM_PROMPT, user_message)
    return answer, sources


# ── Exam Question Solver ───────────────────────────────────────────────────────

EXAM_SYSTEM_PROMPT = """\
You are an expert academic tutor who helps students solve university exam questions.

You will be given:
1. An EXAM QUESTION (possibly with mark allocation, e.g. "8 marks").
2. CONTEXT from the student's uploaded study materials.

Your task:
- Provide a complete, exam-ready answer structured for the marks allocated.
- If marks are mentioned (e.g. 8 marks), aim for ~1 key point per mark.
- Use the context as the primary source; supplement with your knowledge if needed.
- Structure your answer with clear headings/sub-sections.
- Include:
  • Definitions of key terms
  • Explanations with examples
  • Diagrams described in text (e.g., "Diagram: ER model showing...")
  • Any relevant formulas or algorithms
- Use Markdown formatting for readability.
- End with a concise "Exam Tip" if relevant.

Write as if this answer would score full marks in a university exam.
"""


def solve_question(query: str, k: int = 6) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks and generate a model exam answer.

    Returns:
        answer (str): Markdown-formatted model answer.
        sources (list): The retrieved chunk dicts used as context.
    """
    sources = search(query, k=k)

    if not sources:
        return (
            "⚠️ No relevant content found in your uploaded documents. "
            "Please upload study materials related to this question first.",
            [],
        )

    context = _build_context(sources)

    user_message = f"""EXAM QUESTION:
{query}

CONTEXT FROM STUDY MATERIALS:
{context}

Please provide a complete, exam-ready answer for the above question."""

    answer = _call_gemini(EXAM_SYSTEM_PROMPT, user_message)
    return answer, sources