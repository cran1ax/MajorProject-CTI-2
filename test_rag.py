from rag_engine import build_vector_store, ask_rag

with open("knowledge.txt", "r", encoding="utf-8") as f:
    knowledge_data = f.read()

vector_db = build_vector_store(knowledge_data)

answer = ask_rag(vector_db, "What is phishing?")
print("\nRAG OUTPUT:\n")
print(answer)
