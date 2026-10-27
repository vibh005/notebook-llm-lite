
import logging
import math
import time

import pymupdf

from services.text_chunker import chunk_text
from services.embeddings import generate_embeddings


logger = logging.getLogger(__name__)


def is_quota_exhausted(error_message: str) -> bool:
    """
    Identify errors that indicate an exhausted API quota.
    These errors should not be retried automatically.
    """

    message = error_message.lower()

    quota_indicators = [
        "per day",
        "daily limit",
        "daily quota",
        "quota has been exhausted",
        "quota exceeded",
        "exceeded your current quota",
        "generate_content_free_tier_requests",
        "requests per day limit exceeded",
    ]

    return any(
        indicator in message
        for indicator in quota_indicators
    )


def is_temporary_error(error_message: str) -> bool:
    """
    Identify errors that may resolve after waiting.
    """

    message = error_message.lower()

    temporary_indicators = [
        "503",
        "service unavailable",
        "temporarily unavailable",
        "429",
        "resource_exhausted",
        "500",
        "502",
        "504",
        "timeout",
        "timed out",
        "connection reset",
        "connection error",
    ]

    return any(
        indicator in message
        for indicator in temporary_indicators
    )


def retry_gemini_operation(
    operation,
    operation_name: str,
    max_attempts: int = 3,
):
    """
    Retry temporary Gemini failures using exponential backoff.

    Daily quota exhaustion is not retried because waiting
    a few seconds will not resolve a daily quota limit.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    for attempt in range(max_attempts):
        try:
            return operation()

        except Exception as error:
            error_message = str(error)

            # Do not retry an exhausted daily quota.
            if is_quota_exhausted(error_message):
                raise RuntimeError(
                    f"{operation_name} stopped because the Gemini "
                    "API quota appears to be exhausted. "
                    "Wait for the quota to reset or check your "
                    "Gemini API quota settings. "
                    f"Details: {error_message}"
                ) from error

            # Do not retry errors that are not recognized
            # as temporary.
            if not is_temporary_error(error_message):
                raise RuntimeError(
                    f"{operation_name} failed: {error_message}"
                ) from error

            # Stop if there are no attempts remaining.
            if attempt == max_attempts - 1:
                raise RuntimeError(
                    f"{operation_name} failed after "
                    f"{max_attempts} attempts. "
                    f"Last error: {error_message}"
                ) from error

            # Exponential backoff: 2 seconds, then 4 seconds.
            wait_seconds = 2 ** (attempt + 1)

            logger.warning(
                "%s temporarily failed. Retrying in %s seconds "
                "(attempt %s/%s).",
                operation_name,
                wait_seconds,
                attempt + 2,
                max_attempts,
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"{operation_name} failed unexpectedly."
    )


def validate_embeddings(
    embeddings: list,
    expected_count: int,
) -> None:
    """
    Check that the embedding response contains valid,
    consistently sized vectors.
    """

    if len(embeddings) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} embeddings, "
            f"but received {len(embeddings)}."
        )

    if not embeddings:
        raise RuntimeError(
            "The embedding service returned no embeddings."
        )

    expected_dimensions = None

    for index, embedding in enumerate(embeddings):
        if not isinstance(embedding, (list, tuple)):
            raise RuntimeError(
                f"Embedding {index} is not a list or tuple."
            )

        if not embedding:
            raise RuntimeError(
                f"Embedding {index} is empty."
            )

        if expected_dimensions is None:
            expected_dimensions = len(embedding)

        elif len(embedding) != expected_dimensions:
            raise RuntimeError(
                f"Embedding {index} has {len(embedding)} dimensions; "
                f"expected {expected_dimensions}."
            )

        for value in embedding:
            if not isinstance(value, (int, float)):
                raise RuntimeError(
                    f"Embedding {index} contains a non-numeric value."
                )

            if not math.isfinite(value):
                raise RuntimeError(
                    f"Embedding {index} contains an invalid "
                    "infinite or NaN value."
                )


def process_pdf(pdf_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract PDF content, analyze text-poor pages using Gemini Vision,
    create chunks, generate embeddings, and return document records.
    """

    if not pdf_bytes:
        raise ValueError(
            "The uploaded PDF is empty."
        )

    if not filename or not filename.strip():
        raise ValueError(
            "A valid filename is required."
        )

    records = []

    # Step 1: Open and validate the PDF.
    try:
        with pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        ) as pdf:

            if len(pdf) == 0:
                raise ValueError(
                    "The uploaded PDF contains no pages."
                )

            # Step 2: Extract content from each page.
            for page_number, page in enumerate(pdf, start=1):

                page_text = page.get_text("text").strip()

                logger.info(
                    "Processing %s, page %s. Initially extracted "
                    "%s characters.",
                    filename,
                    page_number,
                    len(page_text),
                )

                # Step 3: Use Gemini Vision for pages with
                # very little extractable text.
                if len(page_text) < 30:

                    from services.vision_processor import (
                        extract_page_content,
                    )

                    logger.info(
                        "Using Gemini Vision for %s, page %s.",
                        filename,
                        page_number,
                    )

                    try:
                        page_text = retry_gemini_operation(
                            operation=lambda: extract_page_content(
                                pdf_bytes=pdf_bytes,
                                page_number=page_number,
                            ),
                            operation_name=(
                                f"Gemini Vision processing for "
                                f"{filename}, page {page_number}"
                            ),
                        )

                    
                    except Exception as error:
                        logger.warning(
           "Skipping page %s in %s because Vision extraction failed: %s",
           page_number,
           filename,
           error,
       )
                        continue


                # Step 4: Skip pages with no extractable content.
                if not page_text or not page_text.strip():
                    logger.warning(
                        "Skipping empty page %s in %s.",
                        page_number,
                        filename,
                    )
                    continue

                # Step 5: Split page content into overlapping chunks.
                chunks = chunk_text(
                    page_text,
                    chunk_size=200,
                    overlap=40,
                )

                # Step 6: Identify the page heading.
                heading = next(
                    (
                        line.strip()
                        for line in page_text.splitlines()
                        if line.strip()
                    ),
                    filename,
                )

                # Step 7: Create one record for each chunk.
                for chunk_index, chunk in enumerate(chunks):

                    searchable_text = (
                        f"Document: {filename}\n"
                        f"Section: {heading}\n"
                        f"Content: {chunk}"
                    )

                    records.append({
                        "text": chunk,
                        "searchable_text": searchable_text,
                        "filename": filename,
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                    })

    except (ValueError, RuntimeError):
        raise

    except Exception as error:
        raise RuntimeError(
            f"Could not read or process PDF '{filename}': {error}"
        ) from error

    # Step 8: Ensure that searchable content was extracted.
    if not records:
        raise ValueError(
            f"No searchable content could be extracted from "
            f"'{filename}'. The PDF may be blank, image-only, "
            "or contain pages that Gemini Vision could not analyze."
        )

    # Step 9: Generate embeddings for the chunks.
    logger.info(
        "Generating embeddings for %s chunks from %s.",
        len(records),
        filename,
    )

    texts = [
        record["searchable_text"]
        for record in records
    ]

    embeddings = retry_gemini_operation(
        operation=lambda: generate_embeddings(texts),
        operation_name=f"Gemini embedding generation for {filename}",
    )

    # Step 10: Validate the embedding response.
    validate_embeddings(
        embeddings=embeddings,
        expected_count=len(records),
    )

    # Step 11: Attach each embedding to its corresponding record.
    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding

    logger.info(
        "Successfully processed %s: %s chunks generated.",
        filename,
        len(records),
    )

    return records