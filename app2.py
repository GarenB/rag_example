import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

load_dotenv()

OLLAMA = os.getenv("OLLAMA")
EMBED_MODEL = os.getenv("EMBED_MODEL")
CHAT_MODEL = os.getenv("CHAT_MODEL")

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA) # initializing embeddings
llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA, temperature=0)

chunks = []
for file in Path("docs").glob("*.md"):
    text = file.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    for p in paragraphs:
        chunks.append(Document(page_content=p, metadata={"source": file.name}))

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
    normalize_L2=True
)

print(f"Indexes {len(chunks)} chunks\n")

def retrieve(question, number_of_results=5):
    return vectorstore.similarity_search(question, k=number_of_results)

def ask_llm(question, context):
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know".

Context:
{context}

Question: {question}
"""
    response = llm.invoke(prompt)
    return response.content

while True:
    question = input("Ask: ").strip()
    
    relevant_docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    
    print("\nRetrieved context:")
    print(context)
    
    answer = ask_llm(question, context)
    
    print("\nAnswer:")
    print(answer)
    print()
