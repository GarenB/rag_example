import json
from sse_starlette.sse import EventSourceResponse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.rag.pipeline import RAGPipeline

load_dotenv()

rag_pipeline = RAGPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing RAG pipeline...")
    rag_pipeline.initialize()
    print("RAG pipeline ready!")
    yield
    print("Shutting down application...")


app = FastAPI(
    title="RAG Knowledge Assistant API",
    description="A document-based question answering API powered by Ollama and FAISS.",
    version="1.0.0",
    lifespan=lifespan
)

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

@app.get("/")
def home():
    return {
        "message": "RAG Knowledge Assistant API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        response = rag_pipeline.ask(request.question)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/ask/stream")
async def ask_question_stream(request: QuestionRequest):
    async def event_generator():
        try:
            async for event in rag_pipeline.ask_stream(request.question):
                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"])
                }
        except Exception:
            yield {
                "event": "error",
                "data": json.dumps({
                    "message": "An error occurred during generation."
                })
            }
    return EventSourceResponse(event_generator())
