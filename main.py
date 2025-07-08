# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv
import os

# ── Load environment ──────────────────────────────
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ── FastAPI app ──────────────────────────────────
app = FastAPI()

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

# ── Route ────────────────────────────────────────
@app.post("/qa-embed", response_model=QAEmbedResponse)
def embed_question_and_answer(req: QAEmbedRequest):
    # 1) Create embeddings
    try:
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
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # 2) Insert both embeddings into Supabase
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
        raise HTTPException(status_code=500, detail=f"Supabase insert error: {e}")

    return {
        "status": "ok",
        "question_embedding": question_embedding,
        "answer_embedding": answer_embedding
    }
