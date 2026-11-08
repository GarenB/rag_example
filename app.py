import requests
import faiss
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OLLAMA = os.getenv("OLLAMA")
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

def embed(texts, is_query=False):

    prefix = "search_query: " if is_query else "search_document: "
    inputs = [prefix + text for text in texts]

    r = requests.post(
        f"{OLLAMA}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": inputs
        },
        timeout=120
    )
    r.raise_for_status()

    embeddings = np.array(r.json()["embeddings"], dtype=np.float32)
    embeddings /= (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12) # normalize embeddings
    return embeddings

def ask_llm(prompt):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}]
    })
    return r.json()["message"]["content"]
    
chunks = []
for file in Path("docs").glob("*.md"):
    chunks += [p.strip() for p in file.read_text(encoding="utf-8").split("\n\n") if p.strip()]

chunk_vectors = embed(chunks, is_query=True)
print(f"Indexed {len(chunks)} chunks\n")

dim = chunk_vectors.shape[1] # create FAISS index
index = faiss.IndexFlatIP(dim) # IndexFlatIP = cosine similary / IndexFlatIP = euclidean dist
index.add(chunk_vectors)

def retrieve(question, k=5):
    q = embed([question], is_query=True)
    q = np.asarray(q, dtype=np.float32).reshape(1, -1)
    scores, indices = index.search(q, min(k, index.ntotal))
    # scores = chunk_vectors @ q / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(q))
    # return [chunks[i] for i in np.argsort(scores)[::-1][:k]]
    return [chunks[i] for i in indices[0] if i != -1]

while True:
    question = input("Ask: ")
    context = "\n\n".join(retrieve(question))
    print("context: ", context)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {question}"""
    print("\n" + ask_llm(prompt) + "\n")
