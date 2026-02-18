import os
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://aipipe.org/openai/v1"

app = FastAPI(title="StreamText Inc. LLM Handler")

# --- CORS CONFIGURATION ---
# This allows the assignment platform (running on any domain) to hit your API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Initialize OpenAI Client
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL
)

class StreamRequest(BaseModel):
    prompt: str
    stream: bool = True

async def llm_streamer(prompt: str) -> AsyncGenerator[str, None]:
    """
    Generator that calls OpenAI API with stream=True and yields
    data in SSE format.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data analyst. Generate a 188-word report about space exploration including data analysis and recommendations. Ensure the length is at least 752 characters."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            timeout=10.0
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                data = {
                    "choices": [{
                        "delta": {"content": content}
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"

@app.post("/generate")
async def stream_content(request: StreamRequest):
    if not request.stream:
        raise HTTPException(status_code=400, detail="This endpoint only supports streaming (stream=true)")

    return StreamingResponse(
        llm_streamer(request.prompt),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    # Railway-ready port handling
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)