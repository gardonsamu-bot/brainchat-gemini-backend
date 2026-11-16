
import os
from typing import List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI(title="BrainChat Gemini Backend")

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
APP_TOKEN = os.getenv("APP_ACCESS_TOKEN", "")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = Field(default_factory=list)

@app.get("/")
def home():
    return {"status": "online", "app": "BrainChat"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(
    data: ChatRequest,
    x_app_token: Optional[str] = Header(default=None, alias="X-App-Token")
):
    if not API_KEY:
        raise HTTPException(status_code=503, detail="Gemini API key not configured")

    if APP_TOKEN and x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid app token")

    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        messages = [{
            "role": "system",
            "content": "You are BrainChat, a helpful AI assistant. Answer clearly and accurately."
        }]

        for item in data.history[-20:]:
            messages.append({
                "role": item.role,
                "content": item.content
            })

        messages.append({
            "role": "user",
            "content": data.message
        })

        result = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=2048
        )

        return {"answer": result.choices[0].message.content}

    except Exception as error:
        print("Gemini error:", str(error))
        raise HTTPException(status_code=502, detail="Gemini request failed")
