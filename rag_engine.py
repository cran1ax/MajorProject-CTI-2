from langchain_huggingface import HuggingFaceEmbeddings
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
    model="distilgpt2"
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

    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    result = llm(prompt, truncation=True, max_new_tokens=50, num_return_sequences=1, clean_up_tokenization_spaces=True)
    full_text = result[0]["generated_text"]

    # Extract only the answer part
    if "Answer:" in full_text:
        answer = full_text.split("Answer:")[-1].strip()
    else:
        answer = full_text[len(prompt):].strip()

    return answer
