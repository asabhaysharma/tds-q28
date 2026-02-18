import os
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

# Configuration from environment and question prompt
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="StreamText Inc. LLM Handler")

# Initialize the OpenAI client
client = AsyncOpenAI(
api_key=OPENAI_API_KEY, 
# REPLACE THIS URL with the one from your assignment instructions
base_url="https://aipipe.org/openai/v1" 
)

class StreamRequest(BaseModel):
    prompt: str
    stream: bool = True

async def llm_streamer(prompt: str) -> AsyncGenerator[str, None]:
    """
    Generator that calls OpenAI API with stream=True and yields
    data in SSE format: data: {"choices": [{"delta": {"content": "..."}}]}
    """
    try:
        # Requesting 188-word space report (~250-300 tokens)
        response = await client.chat.completions.create(
            model="gpt-4o",  # or your preferred model
            messages=[
                {"role": "system", "content": "You are a data analyst. Generate a 188-word report about space exploration including data analysis and recommendations. Ensure the length is at least 752 characters."},
                {"role": "user", "content": prompt}
            ],
            stream=True,
            timeout=10.0 # Handle potential timeouts
        )

        async for chunk in response:
            # Extract content delta
            delta_content = chunk.choices[0].delta.content
            
            if delta_content:
                # Format as SSE
                data = {
                    "choices": [{
                        "delta": {"content": delta_content}
                    }]
                }
                yield f"data: {json.dumps(data)}\n\n"

        # Signal completion
        yield "data: [DONE]\n\n"

    except Exception as e:
        # Graceful error handling in the stream
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
    finally:
        # Closing logic is handled by the AsyncOpenAI generator and FastAPI connection
        pass

@app.post("/generate")
async def stream_content(request: StreamRequest):
    """
    POST endpoint to deliver real-time content.
    """
    if not request.stream:
        raise HTTPException(status_code=400, detail="This endpoint only supports streaming (stream=true)")

    return StreamingResponse(
        llm_streamer(request.prompt),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    import os

    # Railway provides the port via the PORT environment variable
    # We default to 8000 for local development
    port = int(os.getenv("PORT", 8000))
    
    # host must be "0.0.0.0" to be accessible externally
    uvicorn.run(app, host="0.0.0.0", port=port)