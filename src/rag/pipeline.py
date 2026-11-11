
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_classic.chains import create_retrieval_chain


class RAGPipeline:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA")
        self.embed_model = os.getenv("EMBED_MODEL")
        self.chat_model = os.getenv("CHAT_MODEL")
        
        self.embeddings = OllamaEmbeddings(
            model=self.embed_model,
            base_url=self.ollama_url
        )
        
        self.llm = ChatOllama(
            model=self.chat_model,
            base_url=self.ollama_url,
            temperature=0
        )
        self.retriever = None
        self.rag_chain = None

    def load_documents(self):
        chunks = []
        for file in Path("docs").glob("*.md"):
            text = file.read_text(encoding="utf-8")

            paragraphs = [
                p.strip()
                for p in text.split("\n\n")
                if p.strip()
            ]

            for paragraph in paragraphs:
                chunks.append(
                    Document(
                        page_content=paragraph,
                        metadata={"source": file.name}
                    )
                )
        return chunks

    def build_vectorstore(self):
        chunks = self.load_documents()

        if not chunks:
            raise ValueError("No documents found in docs directory.")

        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            normalize_L2=True
        )

        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        print(f"Indexed {len(chunks)} chunks")

    def build_chain(self):
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentences maximum. "
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(
            self.llm,
            prompt
        )

        self.rag_chain = create_retrieval_chain(
            self.retriever,
            question_answer_chain
        )

    def initialize(self):
        self.build_vectorstore()
        self.build_chain()

    def ask(self, question: str):
        if self.rag_chain is None:
            raise RuntimeError("RAG pipeline is not initialized.")

        response = self.rag_chain.invoke({
            "input": question
        })

        sources = []

        for doc in response["context"]:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content
            })

        return {
            "question": question,
            "answer": response["answer"],
            "sources": sources
        }

    async def ask_stream(self, question: str):
        if self.rag_chain is None:
            raise RuntimeError("RAG pipeline is not initialized.")

        async for chunk in self.rag_chain.astream({
            "input": question
        }):

            # Retrieved documents
            context = chunk.get("context")

            if context:
                sources = [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "content": doc.page_content
                    }
                    for doc in context
                ]

                yield {
                    "event": "sources",
                    "data": sources
                }

            # Generated answer
            answer_chunk = chunk.get("answer")

            if answer_chunk:
                yield {
                    "event": "token",
                    "data": answer_chunk
                }

        yield {
            "event": "done",
            "data": "[DONE]"
        }


