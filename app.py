from fastapi import FastAPI
import chromadb
import ollama

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    answer = ollama.generate(
        model="tinyllama",
        prompt=f"You are a helpful assistant. Using only the context below, answer the question directly and briefly. Do not repeat the question.\n\nContext: {context}\n\nQuestion: {q}\n\nAnswer:"
    )

    # Strip any leftover prompt leakage
    response = answer["response"].strip()
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    return {"answer": response}