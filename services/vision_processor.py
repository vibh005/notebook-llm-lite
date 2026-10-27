
import os

import pymupdf
from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load environment variables from .env.
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=api_key)

# Gemini model used for PDF page analysis.
MODEL_NAME = "gemini-3.6-flash"


VISION_PROMPT = """
Analyze this PDF page carefully.

Extract readable text, headings, bullet points, and labels.

If the page contains a diagram:
- Identify its main components.
- Explain the role of each component.
- Describe the connections and flow between components.
- Preserve technical terms and relationships accurately.

Return clear, structured text.
Do not invent information that is not visible on the page.
"""


def extract_page_content(
    pdf_bytes: bytes,
    page_number: int,
) -> str:
    """
    Extract text and visual information from one PDF page.

    The document pipeline handles retries and classifies API errors.
    """

    if not pdf_bytes:
        raise ValueError(
            "Cannot analyze an empty PDF."
        )

    if not isinstance(page_number, int) or isinstance(page_number, bool):
        raise ValueError(
            "Page number must be a positive integer."
        )

    if page_number < 1:
        raise ValueError(
            "Page number must be greater than or equal to 1."
        )

    # Step 1: Open the PDF and render the requested page.
    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as pdf:

        if page_number > len(pdf):
            raise ValueError(
                f"Page number must be between 1 and {len(pdf)}."
            )

        page = pdf[page_number - 1]

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(1.5, 1.5),
            alpha=False,
        )

        image_bytes = pixmap.tobytes("png")

    # Step 2: Send the rendered page to Gemini Vision.
    # API exceptions are allowed to propagate so that
    # document_pipeline.py can decide whether to retry.
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            VISION_PROMPT,
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            ),
        ],
    )

    # Step 3: Validate Gemini's response.
    extracted_text = response.text

    if not extracted_text or not extracted_text.strip():
        raise RuntimeError(
            f"Gemini returned no usable visual analysis "
            f"for page {page_number}."
        )

    return extracted_text.strip()
