
import os
from uuid import uuid5, NAMESPACE_URL

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_URL or QDRANT_API_KEY is missing. "
        "Check your .env file."
    )

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30,
)

COLLECTION_NAME = "notebook_documents"
VECTOR_SIZE = 3072


def test_connection() -> bool:
    client.get_collections()
    return True


def create_collection() -> None:
    existing = [
        collection.name
        for collection in client.get_collections().collections
    ]

    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )

    print(f"Collection '{COLLECTION_NAME}' created.")


def create_payload_indexes() -> None:
    """Create indexes used to filter documents by owner and filename."""

    for field in ("user_id", "filename"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    print("User ID and filename indexes are ready.")


def delete_document(user_id: str, filename: str) -> None:
    """Delete a document belonging to this user only."""

    if not user_id or not filename:
        raise ValueError("User ID and filename are required.")

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(value=filename),
                    ),
                ]
            )
        ),
        wait=True,
    )

    print(f"Deleted '{filename}' for the specified user.")


def store_document(user_id: str, records: list[dict]) -> None:
    """Store document chunks under the authenticated user's ID."""

    if not user_id:
        raise ValueError("User ID is required.")

    if not records:
        raise ValueError("No document records were provided.")

    points = []

    for record in records:
        point_id = str(
            uuid5(
                NAMESPACE_URL,
                (
                    f"{user_id}:"
                    f"{record['filename']}:"
                    f"{record['page_number']}:"
                    f"{record['chunk_index']}"
                ),
            )
        )

        embedding = record["embedding"]

        if len(embedding) != VECTOR_SIZE:
            raise ValueError(
                f"Expected {VECTOR_SIZE} vector values, "
                f"but received {len(embedding)}."
            )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "text": record["text"],
                    "searchable_text": record["searchable_text"],
                    "filename": record["filename"],
                    "page_number": record["page_number"],
                    "chunk_index": record["chunk_index"],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    print(f"Stored {len(points)} chunks for the specified user.")


def _cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = sum(a * a for a in vector_a) ** 0.5
    magnitude_b = sum(b * b for b in vector_b) ** 0.5

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def _select_with_mmr(
    candidates: list,
    limit: int,
    query_vector: list[float],
    lambda_param: float = 0.7,
) -> list:
    selected = []
    remaining = list(candidates)

    while remaining and len(selected) < limit:
        best_candidate = None
        best_mmr_score = float("-inf")

        for candidate in remaining:
            relevance = candidate.score

            if selected:
                redundancy = max(
                    _cosine_similarity(
                        candidate.vector,
                        chosen.vector,
                    )
                    for chosen in selected
                )
            else:
                redundancy = 0.0

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * redundancy
            )

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate = candidate

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def search_documents(
    user_id: str,
    question: str,
    limit: int = 5,
    filenames: list[str] | None = None,
) -> list[dict]:
    """Search only documents belonging to the specified user."""

    from services.embeddings import generate_embedding

    if not user_id:
        raise ValueError("User ID is required.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if limit <= 0:
        raise ValueError("Limit must be greater than zero.")

    query_vector = generate_embedding(question)

    if len(query_vector) != VECTOR_SIZE:
        raise ValueError(
            f"Expected a query embedding with {VECTOR_SIZE} dimensions, "
            f"but received {len(query_vector)}."
        )

    conditions = [
        models.FieldCondition(
            key="user_id",
            match=models.MatchValue(value=user_id),
        )
    ]

    if filenames:
        conditions.append(
            models.FieldCondition(
                key="filename",
                match=models.MatchAny(any=filenames),
            )
        )

    query_filter = models.Filter(must=conditions)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=max(limit * 4, 20),
        with_payload=True,
        with_vectors=True,
        score_threshold=0.50,
    )

    candidates = response.points

    if not candidates:
        return []

    selected_chunks = _select_with_mmr(
        candidates=candidates,
        limit=limit,
        query_vector=query_vector,
        lambda_param=0.7,
    )

    results = []

    for point in selected_chunks:
        payload = point.payload or {}

        results.append(
            {
                "text": payload.get("text", ""),
                "filename": payload.get("filename", ""),
                "page_number": payload.get("page_number"),
                "score": point.score,
            }
        )

    return results