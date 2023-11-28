from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from pydantic import BaseModel

import numpy as np
import io

app = FastAPI()

# # Allow all requests (optional, good for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict(original_language: str = None, to_translate_language: str = None, audio_file: UploadFile = File(...)):
    if original_language is None or to_translate_language is None:
        return {"error": "original_language and to_translate_language are required query parameters"}

    print(audio_file.filename)
    # Process the audio file and perform translation logic
    original_text = f"Original text in {original_language}: Sample original text."

    # Process audio_file - Replace this section with your actual translation logic
    # For now, just provide placeholder translated text
    translated_text = f"Translated text to {to_translate_language}: Sample translated text."

    response_body = {"original_text": original_text, "translated_text": translated_text}
    return response_body