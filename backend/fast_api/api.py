import io
import os
import torch
import base64

import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from services import speech_to_text, text_to_text, text_to_speech
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from utils.preparation import extract_feature

language_codes = {
    'english': 'en-UK',
    'bengali': 'bn-BN'
}

error_messages  = {
    "en-UK": 'No speech recognized, please try again',
    "bn-BN": 'কোনো বক্তৃতা স্বীকৃত নয়, অনুগ্রহ করে আবার চেষ্টা করুন'
}

app = FastAPI()
app.state.model = load_model('./model/model.h5')
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
    audio_content = await parse_audio(audio_file)

    original_language = language_codes.get(original_language)
    to_translate_language = language_codes.get(to_translate_language)

    first_step = speech_to_text.speech_to_text(audio_content, original_language)

    if first_step == "":
        return await return_error_message(original_language)

    gender = await get_gender_prediction()

    second_step = text_to_text.text_to_text(first_step, original_language, to_translate_language)

    third_step =  await text_to_speech.text_to_speech(second_step, to_translate_language, gender)

    # Encode the binary audio data as a Base64 string
    encoded_audio = base64.b64encode(third_step).decode('utf-8')

    # Prepare the response data
    response_data = {
        "first_step": first_step,
        "second_step": second_step,
        "third_step": encoded_audio
    }

    return JSONResponse(content=response_data)

async def parse_audio(audio_file: UploadFile = File(...)):
    audio_content = await audio_file.read()
    input_audio_path = 'input.wav'
    input_audio_bytes = io.BytesIO(audio_content)
    with open(input_audio_path, 'wb') as file:
        file.write(audio_content)
    audio = AudioSegment.from_file(input_audio_path)

    new_sample_rate = 44100
    audio = audio.set_frame_rate(new_sample_rate)
    audio.export("input.wav", format="wav", parameters=["-acodec", "pcm_s16le"])  # for 16-bit depth
    return audio_content

async def get_gender_prediction():
    features = extract_feature("input.wav").reshape(1, -1)
    male_prob = app.state.model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    return gender

async def return_error_message(original_language):
    print(error_messages[original_language], original_language)
    third_step =  await text_to_speech.text_to_speech(error_messages[original_language], original_language, "male")

    encoded_audio = base64.b64encode(third_step).decode('utf-8')
    return {
        'first_step': '',
        'second_step': '',
        'third_step':  encoded_audio
    }