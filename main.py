from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from supabase import create_client
from openai import OpenAI
import os

# ── FastAPI app ──────────────────────────────────
app = FastAPI()

# ✅ Add CORS middleware to allow frontend to call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your local dev URL or production domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ──────────────────────────────
class QAEmbedRequest(BaseModel):
    question: str
    question_id: str
    answer: str
    answer_id: str
    session_id: str

class QAEmbedResponse(BaseModel):
    status: str
    question_embedding: List[float]
    answer_embedding: List[float]

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/qa-embed", response_model=QAEmbedResponse)
def embed_question_and_answer(req: QAEmbedRequest):
    # Get env vars inside the request to ensure they're there
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

        question_resp = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=req.question
        )
        answer_resp = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=req.answer
        )

        question_embedding = question_resp.data[0].embedding
        answer_embedding = answer_resp.data[0].embedding

    except Exception as e:
        print("Embedding error:", e)
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    try:
        supabase.table("ManagementAI_Embeddings").insert([
            {
                "MessageID": req.question_id,
                "SessionID": req.session_id,
                "Embedding": question_embedding
            },
            {
                "MessageID": req.answer_id,
                "SessionID": req.session_id,
                "Embedding": answer_embedding
            }
        ]).execute()

    except Exception as e:
        print("Supabase insert error:", e)
        raise HTTPException(status_code=500, detail=f"Supabase insert error: {e}")

    return {
        "status": "ok",
        "question_embedding": question_embedding,
        "answer_embedding": answer_embedding
    }
