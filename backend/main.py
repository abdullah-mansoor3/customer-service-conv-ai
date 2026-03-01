from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import certifi

app = FastAPI(title="Llama.cpp Backend Proxy")

# Allow all origins for seamless frontend integration during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant. Always respond concisely and stop generating text once you have answered the question."
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Proxies a chat request to the local llama-server.
    By using the /v1/chat/completions endpoint with a 'messages' array,
    llama.cpp will automatically format this using the model's native chat template
    (e.g., ChatML, Llama-3, etc.), ensuring full compatibility regardless of the model loaded!
    """
    
    # We construct the messages array which llama.cpp natively translates
    # into the correct chat template (like ChatML or Llama-3 limits).
    payload = {
        "messages": [
            {
                "role": "system",
                "content": request.system_prompt
            },
            {
                "role": "user",
                "content": request.message
            }
        ],
        "temperature": request.temperature,
        "max_tokens": -1,  # Let it generate naturally up to limit
        "stream": False    # For a simple stateless backend, we await the full response first
    }

    try:
        # Provide extended timeout for LLM generation
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            # Extract the raw text from the OpenAI-compatible response format
            generated_text = data['choices'][0]['message']['content']
            
            return ChatResponse(response=generated_text)
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"LLM Server Error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail="Could not connect to the local Llama Server. Is it running?")

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "llama_server": LLAMA_SERVER_URL}
