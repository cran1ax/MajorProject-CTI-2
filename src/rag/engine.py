"""
RAG Engine for Cybersecurity Threat Intelligence Bot.

Architecture: ChromaDB (vector store) + llama.cpp (local quantized LLM)
Flow: User query + ML prediction → ChromaDB retrieval → Prompt assembly → LLM generation
"""

import os
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — all relative to the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
DEFAULT_KNOWLEDGE_PATH = str(PROJECT_ROOT / "src" / "rag" / "knowledge.txt")
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q5_k_m.gguf")

COLLECTION_NAME = "cybersecurity_kb"

# ---------------------------------------------------------------------------
# 1.  ChromaDB initialisation (persistent on-disk collection)
# ---------------------------------------------------------------------------

_chroma_client: Optional[chromadb.ClientAPI] = None
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2 …")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def init_chroma(persist_dir: str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    """
    Initialise (or reconnect to) a persistent ChromaDB client.

    Returns
    -------
    chromadb.ClientAPI
        A ChromaDB client backed by on-disk storage at *persist_dir*.
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    os.makedirs(persist_dir, exist_ok=True)
    logger.info("Initialising ChromaDB at %s", persist_dir)

    _chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    return _chroma_client


def get_or_create_collection(
    client: Optional[chromadb.ClientAPI] = None,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Return the KB collection, creating it if it does not yet exist."""
    if client is None:
        client = init_chroma()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# 2.  Document ingestion with RecursiveCharacterTextSplitter
# ---------------------------------------------------------------------------

def ingest_documents(
    text: str,
    collection: Optional[chromadb.Collection] = None,
    source_label: str = "knowledge_base",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    """
    Split *text* into overlapping chunks and upsert them into ChromaDB.

    Parameters
    ----------
    text : str
        Raw document text (Markdown, plain-text, etc.).
    collection : chromadb.Collection, optional
        Target collection.  Uses the default KB collection when *None*.
    source_label : str
        Metadata tag attached to every chunk for filtering / provenance.
    chunk_size : int
        Maximum characters per chunk (≈ tokens for English text).
    chunk_overlap : int
        Overlap between consecutive chunks to avoid losing boundary context.

    Returns
    -------
    int
        Number of chunks ingested.
    """
    if collection is None:
        collection = get_or_create_collection()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_text(text)
    if not chunks:
        logger.warning("No chunks produced from the input text.")
        return 0

    # Embed the chunks
    model = _get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    # Generate stable IDs so re-ingestion is idempotent (upsert)
    ids = [f"{source_label}__chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_label, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Ingested %d chunks from '%s' into ChromaDB.", len(chunks), source_label)
    return len(chunks)


def ingest_knowledge_file(
    filepath: str = DEFAULT_KNOWLEDGE_PATH,
    collection: Optional[chromadb.Collection] = None,
) -> int:
    """Convenience wrapper: read a text file and ingest its contents."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge file not found: {filepath}")

    text = path.read_text(encoding="utf-8")
    source = path.stem  # e.g. "knowledge"
    return ingest_documents(text, collection=collection, source_label=source)


# ---------------------------------------------------------------------------
# 3.  Local LLM initialisation via llama-cpp-python
# ---------------------------------------------------------------------------

_llm_instance = None  # lazy singleton


class LLMLoadError(RuntimeError):
    """Raised when the GGUF model file cannot be loaded."""


def init_llm(
    model_path: str = DEFAULT_MODEL_PATH,
    n_ctx: int = 2048,
    n_threads: int = 4,
    n_gpu_layers: int = 0,
):
    """
    Load a quantized GGUF model via llama-cpp-python.

    Parameters
    ----------
    model_path : str
        Absolute or relative path to the ``.gguf`` model file.
    n_ctx : int
        Context window size in tokens.
    n_threads : int
        Number of CPU threads for inference.
    n_gpu_layers : int
        Layers to offload to GPU (0 = CPU-only).

    Raises
    ------
    LLMLoadError
        If the model file is missing or llama-cpp-python fails to load it.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    resolved = Path(model_path).resolve()
    if not resolved.is_file():
        raise LLMLoadError(
            f"GGUF model file not found at: {resolved}\n"
            f"  → Download Qwen2.5-3B-Instruct GGUF (~1.8 GB) from:\n"
            f"    https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF\n"
            f"    Place the Q4_K_M .gguf file in the 'models/' directory."
        )

    try:
        from llama_cpp import Llama  # import here to fail gracefully if not installed

        logger.info("Loading LLM from %s  (n_ctx=%d, threads=%d, gpu_layers=%d) …",
                     resolved, n_ctx, n_threads, n_gpu_layers)
        _llm_instance = Llama(
            model_path=str(resolved),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info("LLM loaded successfully.")
        return _llm_instance

    except ImportError:
        raise LLMLoadError(
            "llama-cpp-python is not installed.\n"
            "  → Run: pip install llama-cpp-python"
        )
    except Exception as exc:
        raise LLMLoadError(f"Failed to load GGUF model: {exc}") from exc


def _generate_text(prompt: str, max_tokens: int = 256) -> str:
    """Run inference on the loaded LLM and return the generated text."""
    llm = init_llm()
    output = llm(prompt, max_tokens=max_tokens, stop=["\n\n\n"], echo=False)
    return output["choices"][0]["text"].strip()


# ---------------------------------------------------------------------------
# 4.  Core inference: generate_threat_explanation()
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
<|im_start|>system
You are a cybersecurity threat analyst. Your job is to explain why a machine-learning
model predicted a certain risk level, using the retrieved intelligence documents below.
Answer concisely. If the retrieved documents do not contain relevant information,
say so rather than guessing.<|im_end|>
<|im_start|>user
## ML Prediction
The model predicts a {prob_pct} probability of HIGH RISK.
Attack: {attack_type} | Industry: {industry} | Country: {country}
Financial loss: ${financial_loss}M | Affected users: {affected_users:,}
Vulnerability score: {vuln_score}/10 | Response time: {response_time}h

## Retrieved Intelligence
{context}

## Question
{user_query}<|im_end|>
<|im_start|>assistant
"""


def _retrieve_context(
    query: str,
    collection: Optional[chromadb.Collection] = None,
    n_results: int = 3,
) -> str:
    """
    Embed *query* and retrieve the top-k most relevant chunks from ChromaDB.

    Returns a single string with the chunks separated by blank lines.
    """
    if collection is None:
        collection = get_or_create_collection()

    model = _get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    if not documents:
        return "(No relevant documents found in the knowledge base.)"

    # Format each chunk with its source metadata
    parts = []
    metadatas = results.get("metadatas", [[]])[0]
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        source = meta.get("source", "unknown") if meta else "unknown"
        parts.append(f"[{i}] (source: {source})\n{doc}")

    return "\n\n".join(parts)


def generate_threat_explanation(
    prediction_prob: float,
    features: dict,
    user_query: str,
    collection: Optional[chromadb.Collection] = None,
    max_tokens: int = 256,
) -> str:
    """
    End-to-end RAG pipeline: ML prediction + ChromaDB retrieval + LLM generation.

    Parameters
    ----------
    prediction_prob : float
        The ML model's predicted probability of high risk (0.0 – 1.0).
    features : dict
        Must contain keys:
            attack_type, industry, country,
            financial_loss, affected_users,
            vulnerability_score, response_time
    user_query : str
        The human's question (e.g. "Why is this high risk?").
    collection : chromadb.Collection, optional
        ChromaDB collection to query.  Uses default KB if *None*.
    max_tokens : int
        Maximum tokens for the LLM response.

    Returns
    -------
    str
        The LLM-generated explanation grounded in retrieved documents.

    Raises
    ------
    LLMLoadError
        If the GGUF model file is missing or cannot be loaded.
    """
    # --- Build a retrieval query that combines the prediction with the question ---
    prob_pct = f"{prediction_prob:.0%}"
    retrieval_query = (
        f"{prob_pct} high risk, {features.get('attack_type', 'unknown')} attack "
        f"on {features.get('industry', 'unknown')} sector. {user_query}"
    )

    # --- Retrieve relevant chunks from ChromaDB ---
    context = _retrieve_context(retrieval_query, collection=collection)

    # --- Assemble the prompt ---
    prompt = _PROMPT_TEMPLATE.format(
        prob_pct=prob_pct,
        attack_type=features.get("attack_type", "N/A"),
        industry=features.get("industry", "N/A"),
        country=features.get("country", "N/A"),
        financial_loss=features.get("financial_loss", 0),
        affected_users=features.get("affected_users", 0),
        vuln_score=features.get("vulnerability_score", 0),
        response_time=features.get("response_time", 0),
        context=context,
        user_query=user_query,
    )

    # --- Generate the response ---
    try:
        return _generate_text(prompt, max_tokens=max_tokens)
    except LLMLoadError:
        raise  # propagate cleanly to the UI layer
    except Exception as exc:
        logger.exception("LLM generation failed.")
        return f"[RAG Engine Error] Could not generate explanation: {exc}"


# ---------------------------------------------------------------------------
# CLI quick-test (python -m src.rag.engine)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # 1. Init ChromaDB and ingest the knowledge file
    client = init_chroma()
    collection = get_or_create_collection(client)
    try:
        n = ingest_knowledge_file(collection=collection)
        print(f"✓ Ingested {n} chunks into ChromaDB")
    except FileNotFoundError as e:
        print(f"⚠ {e}")

    # 2. Test retrieval (does not need the LLM)
    test_query = "What is ransomware and how does it spread?"
    print(f"\n--- Retrieval test: '{test_query}' ---")
    ctx = _retrieve_context(test_query, collection=collection)
    print(ctx)

    # 3. Test full generation (requires a GGUF model in models/)
    print("\n--- Full generation test ---")
    try:
        answer = generate_threat_explanation(
            prediction_prob=0.78,
            features={
                "attack_type": "Ransomware",
                "industry": "Healthcare",
                "country": "USA",
                "financial_loss": 120,
                "affected_users": 35000,
                "vulnerability_score": 8.5,
                "response_time": 2.3,
            },
            user_query="Why is this high risk and what should we do?",
            collection=collection,
        )
        print(answer)
    except LLMLoadError as e:
        print(f"⚠ LLM not available (expected if no .gguf file):\n  {e}")
