
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Check your .env file."
    )

client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "openai/gpt-oss-120b"


def generate_answer(question: str, search_results: list[dict]) -> str:
    """
    Generate a clear, source-grounded answer using passages
    retrieved from Qdrant.
    """

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not search_results:
        return (
            "I couldn't find relevant information in your uploaded "
            "documents. Try asking a different question."
        )

    # Prepare document context with source information.
    context_parts = []

    for index, result in enumerate(search_results, start=1):
        context_parts.append(
            f"[Source {index}]\n"
            f"Filename: {result['filename']}\n"
            f"Page: {result['page_number']}\n"
            f"Content: {result['text']}"
        )

    context = "\n\n".join(context_parts)

    # Generate an answer grounded in the retrieved passages.
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are NotebookLM Lite, a document research assistant.\n\n"

                    "RULES:\n"
                    "1. Answer using the provided document context.\n"
                    "2. Explain technical concepts clearly and simply.\n"
                    "3. Combine relevant information from multiple sources "
                    "into one coherent answer.\n"
                    "4. Do not invent facts or make unsupported claims.\n"
                    "5. If the context is insufficient, clearly state "
                    "what information is missing.\n"
                    "6. Cite factual claims using the exact filename "
                    "and page number provided in the context.\n"
                    "7. Never invent filenames, page numbers, or citations.\n"
                    "8. Avoid unnecessary repetition.\n\n"

                    "CITATION FORMAT:\n"
                    "Use citations such as (UNIT5.pdf, p. 38).\n"
                    "At the end, include a Sources section listing the "
                    "relevant filenames and page numbers.\n"
                    "Only list sources that support your answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Document context:\n{context}\n\n"
                    "Answer the question using the provided context."
                ),
            },
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        raise RuntimeError("Groq returned an empty answer.")

    return answer.strip()
