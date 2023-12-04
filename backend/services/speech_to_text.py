from google.cloud import speech

def speech_to_text(audio_content: bytes, original_language) -> str:
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(content=audio_content)

    config = speech.RecognitionConfig(
        language_code=original_language.split("_")[0],
        enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)

    if response.results:
        best_alternative = response.results[0].alternatives[0]
        return best_alternative.transcript
    else:
        return ""
