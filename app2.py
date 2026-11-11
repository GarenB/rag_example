import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

while True:
    query = input("Ask: ").strip()

    response = rag_chain.invoke({"input": query})
    print("\nRetrieved context:")
    
    for doc in response["context"]:
        print(f"\nSource: {doc.metadata.get('source', 'Unknown')}")
        print(doc.page_content)
    
    print("\nAnswer:")
    print(response["answer"])
    print()
