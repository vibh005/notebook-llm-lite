
import math
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load environment variables.
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-embedding-2"

# Process document chunks in smaller groups.
DEFAULT_BATCH_SIZE = 16


def _validate_embedding(
    values: list[float] | None,
    description: str,
) -> list[float]:
    """
    Validate one embedding vector before returning it.
    """

    if not values:
        raise RuntimeError(
            f"{description}: Gemini returned an empty embedding."
        )

    for value in values:
        if not isinstance(value, (int, float)):
            raise RuntimeError(
                f"{description}: embedding contains a non-numeric value."
            )

        if not math.isfinite(value):
            raise RuntimeError(
                f"{description}: embedding contains an invalid "
                "infinite or NaN value."
            )

    return list(values)


def generate_embeddings(
    texts: list[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """
    Generate one embedding for each document text chunk.

    Inputs are processed in batches to avoid sending an
    unnecessarily large list in a single API request.
    """

    if not texts:
        return []

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if any(
        not isinstance(text, str) or not text.strip()
        for text in texts
    ):
        raise ValueError(
            "Every input must be a non-empty string."
        )

    all_embeddings = []
    expected_dimensions = None

    # Process the inputs in batches.
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        contents = [
            types.Content(
                parts=[
                    types.Part.from_text(
                        text=f"title: none | text: {text}"
                    )
                ]
            )
            for text in batch
        ]

        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=contents,
        )

        if not response.embeddings:
            raise RuntimeError(
                f"Gemini returned no embeddings for batch "
                f"starting at input {start + 1}."
            )

        if len(response.embeddings) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings for batch "
                f"starting at input {start + 1}, but Gemini "
                f"returned {len(response.embeddings)}."
            )

        # Validate each returned vector.
        for offset, embedding in enumerate(response.embeddings):
            values = _validate_embedding(
                embedding.values,
                description=(
                    f"Document embedding {start + offset + 1}"
                ),
            )

            if expected_dimensions is None:
                expected_dimensions = len(values)

            elif len(values) != expected_dimensions:
                raise RuntimeError(
                    "Gemini returned inconsistent embedding dimensions. "
                    f"Expected {expected_dimensions}, "
                    f"received {len(values)}."
                )

            all_embeddings.append(values)

    if len(all_embeddings) != len(texts):
        raise RuntimeError(
            f"Expected {len(texts)} embeddings, "
            f"but generated {len(all_embeddings)}."
        )

    return all_embeddings


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single search query.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            "Query text must be a non-empty string."
        )

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=(
            "task: question answering | query: " + text
        ),
    )

    if not response.embeddings:
        raise RuntimeError(
            "Gemini returned no embedding for the search query."
        )

    if len(response.embeddings) != 1:
        raise RuntimeError(
            "Expected one query embedding, but Gemini returned "
            f"{len(response.embeddings)}."
        )

    return _validate_embedding(
        response.embeddings[0].values,
        description="Search query embedding",
    )
