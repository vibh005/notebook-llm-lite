def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    Split text into overlapping chunks of words.

    Args:
        text: The input text.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start += chunk_size - overlap

    return chunks