import os
import sys
import requests
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

# Cosmos DB configuration
COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
COSMOS_KEY = os.environ.get("COSMOS_KEY")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME")

# Embedding service configuration (used to encode the query)
EMBEDDING_ENDPOINT = os.environ.get("embedding_endpoint")
EMBEDDING_DEPLOYMENT = os.environ.get("embedding_deployment")
EMBEDDING_API_KEY = os.environ.get("embedding_api_key")
EMBEDDING_API_VERSION = os.environ.get("embedding_api_version")

# Validate required Cosmos env vars
if not COSMOS_ENDPOINT:
    raise ValueError("COSMOS_ENDPOINT environment variable is not set")
if not DATABASE_NAME:
    raise ValueError("DATABASE_NAME environment variable is not set")
if not CONTAINER_NAME:
    raise ValueError("CONTAINER_NAME environment variable is not set")


def get_cosmos_client(endpoint: str | None, key: str | None = None):
    if not endpoint:
        raise ValueError("COSMOS_ENDPOINT must be provided in environment variables")

    # Try Entra ID (managed identity) first
    try:
        credential = DefaultAzureCredential()
        client = CosmosClient(endpoint, credential=credential)
        _ = list(client.list_databases())
        return client
    except AzureError:
        pass

    # Fallback to key
    if key:
        client = CosmosClient(endpoint, key)
        return client

    raise RuntimeError(
        "Failed to authenticate to Cosmos DB using DefaultAzureCredential and no valid COSMOS_KEY was provided"
    )


def get_request_embedding(text: str) -> list[float] | None:
    """Call embedding endpoint and return the embedding vector or None on failure."""
    if not EMBEDDING_ENDPOINT or not EMBEDDING_DEPLOYMENT or not EMBEDDING_API_KEY or not EMBEDDING_API_VERSION:
        raise ValueError("Embedding endpoint configuration missing. Set EMBEDDING_ENDPOINT, EMBEDDING_DEPLOYMENT, EMBEDDING_API_KEY, EMBEDDING_API_VERSION")

    url = EMBEDDING_ENDPOINT.rstrip("/") + f"/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings?api-version={EMBEDDING_API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": EMBEDDING_API_KEY,
    }
    payload = {"input": text}

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    embedding = data.get("data", [{}])[0].get("embedding")
    return embedding


def _extract_keywords(question: str) -> list[str]:
    """Extract meaningful keywords from the user's question for full-text search."""
    stop_words = {
        "i", "me", "my", "a", "an", "the", "is", "are", "was", "were",
        "do", "does", "did", "have", "has", "had", "am", "be", "been",
        "for", "of", "to", "in", "on", "at", "by", "with", "from",
        "and", "or", "but", "not", "so", "if", "as", "it", "its",
        "can", "could", "would", "should", "will", "shall", "may",
        "what", "which", "who", "how", "where", "when", "that", "this",
        "im", "looking", "want", "need", "find", "search", "get",
        "about", "some", "any", "best", "good", "like", "please",
    }
    words = question.lower().split()
    keywords = []
    for w in words:
        cleaned = w.strip("?.!,'\"").replace("'", "")
        if cleaned and cleaned not in stop_words:
            keywords.append(cleaned)
    return keywords if keywords else [w.strip("?.!,'\"").replace("'", "") for w in words[:3]]


def _build_fulltext_score(field: str, keywords: list[str]) -> str:
    """Build a FullTextScore call with individual string-literal arguments.
    e.g. FullTextScore(c.ProductName, 'deep', 'forest', 'paint')
    """
    escaped = [kw.replace("'", "''") for kw in keywords]
    args = ", ".join(f"'{kw}'" for kw in escaped)
    return f"FullTextScore({field}, {args})"


# Initialize Cosmos client and container
_cosmos_client = get_cosmos_client(COSMOS_ENDPOINT, COSMOS_KEY)
_database = _cosmos_client.get_database_client(DATABASE_NAME)
_container = _database.get_container_client(CONTAINER_NAME)


def product_recommendations(question: str, top_k: int = 8, mode: str = "hybrid"):
    """
    Search for product recommendations using vector, full-text, or hybrid search.

    Input:
        question (str): Natural language user query
        top_k (int): number of nearest neighbors to return
        mode (str): "vector" | "fulltext" | "hybrid" (default: "hybrid")
    Output:
        list of product dicts with product information
        (same structure as original aiSearchTools.py)
    """

    fields = (
        "c.id, c.ProductID, c.ProductName, c.ProductCategory, "
        "c.ProductDescription, c.ImageURL, c.ProductPunchLine, c.Price"
    )

    if mode == "vector":
        # --- Vector-only search (same as original aiSearchTools.py) ---
        query_vector = get_request_embedding(question)
        if query_vector is None:
            raise RuntimeError("Failed to generate query embedding")

        query = (
            f"SELECT {fields} FROM c "
            "ORDER BY VectorDistance(c.request_vector, @vector) "
            "OFFSET 0 LIMIT @top"
        )
        parameters = [
            {"name": "@vector", "value": query_vector},
            {"name": "@top", "value": top_k},
        ]

    elif mode == "fulltext":
        # --- Full-text only search (BM25 keyword scoring) ---
        keywords = _extract_keywords(question)
        fts_desc = _build_fulltext_score("c.ProductDescription", keywords)

        query = (
            f"SELECT TOP @top {fields} FROM c "
            f"ORDER BY RANK {fts_desc}"
        )
        parameters = [
            {"name": "@top", "value": top_k},
        ]

    else:
        # --- Hybrid search (vector + full-text via RRF) ---
        query_vector = get_request_embedding(question)
        if query_vector is None:
            raise RuntimeError("Failed to generate query embedding")

        keywords = _extract_keywords(question)
        fts_name = _build_fulltext_score("c.ProductName", keywords)
        fts_desc = _build_fulltext_score("c.ProductDescription", keywords)

        query = (
            f"SELECT TOP @top {fields} FROM c "
            f"ORDER BY RANK RRF("
            f"    {fts_name}, "
            f"    {fts_desc}, "
            f"    VectorDistance(c.request_vector, @vector)"
            f")"
        )
        parameters = [
            {"name": "@vector", "value": query_vector},
            {"name": "@top", "value": top_k},
        ]

    items = list(_container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
        max_item_count=top_k,
    ))

    # Same output structure as original aiSearchTools.py
    get = dict.get
    response = [
        {
            "id": get(item, "ProductID", None),
            "name": get(item, "ProductName", None),
            "type": get(item, "ProductCategory", None),
            "description": get(item, "ProductDescription", None),
            "imageURL": get(item, "ImageURL", None),
            "punchLine": get(item, "ProductPunchLine", None),
            "price": get(item, "Price", None)
        }
        for item in items
    ]

    return response
