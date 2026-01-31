from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load lightweight local LLM
llm = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=120
)

def build_vector_store(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

def ask_rag(vector_db, question: str) -> str:
    docs = vector_db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the context below to answer the question clearly.

Context:
{context}

Question:
{question}

Answer:
"""
    result = llm(prompt)
    return result[0]["generated_text"]
