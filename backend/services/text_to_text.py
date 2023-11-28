import requests

def tts(text: str, original_language_simplified_code: str, to_translate_language_simplified_code: str):
    translation_url = f'https://api.mymemory.translated.net/get?q={text}&langpair={original_language_simplified_code}|{to_translate_language_simplified_code}&de=tanushri0310003@gmail.com'

    response = requests.get(translation_url)

    if response.status_code == 200:
        response_data = response.json()
        translated_text = response_data.get('responseData', {}).get('translatedText', '')
        if translated_text:
            return translated_text
    raise Exception("Translation not found in the response")