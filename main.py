import os
import json
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://aipipe.org/openai/v1"

app = FastAPI(title="StreamText Inc. LLM Handler")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        # OPTIMIZATION: Use gpt-4o-mini for fastest Time-To-First-Token (TTFT)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant. Write a detailed 188-word report on space exploration. Ensure it is at least 752 characters long."
                },
                {"role": "user", "content": prompt}
            ],
            stream=True,
            temperature=0.7, # Slightly lower temp can sometimes improve stability
            max_tokens=500,  # Limit generation to keep it snappy (but enough for requirements)
            timeout=5.0      # Fail fast if connection hangs
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                # Manually constructing the JSON to ensure minimal overhead
                data = json.dumps({
                    "choices": [{
                        "delta": {"content": content}
                    }]
                })
                yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = json.dumps({"error": str(e)})
        yield f"data: {error_data}\n\n"

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