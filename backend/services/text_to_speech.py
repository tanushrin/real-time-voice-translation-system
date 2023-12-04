import io

from google.cloud import texttospeech
from starlette.responses import StreamingResponse


async def text_to_speech(text: str, language_code: str, gender: str):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    if gender == "male":
        ssml_gender = texttospeech.SsmlVoiceGender.MALE
    else:
        ssml_gender = texttospeech.SsmlVoiceGender.FEMALE

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, ssml_gender=ssml_gender
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    audio_content = response.audio_content

    return audio_content