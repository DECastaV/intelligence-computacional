from openai import OpenAI
import os
from dotenv import load_dotenv

SYSTEM_MESSAGE = "You are a chatbot. You will have a conversation with a user. Be friendly and concise."

if __name__ == "__main__":
    load_dotenv()
    URL = os.environ.get('OPENAI_BASE_URL')
    KEY = os.environ.get('OPENAI_KEY')
    MODEL = os.environ.get('MODEL')

    client = OpenAI(
        base_url=URL,
        api_key=KEY,
    )

    print(f"Chatting with {MODEL} model at {URL}\n")
    
    
    message = input("> ")

    messages = [
                {'role': 'system', 'content': SYSTEM_MESSAGE},
                {'role': 'user', 'content': message},
            ]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        respuesta = response.choices[0].message.content
        messages.append({'role': 'assistant', 'content': respuesta})
        messages.append({'role': 'user', 'content': message})
        print(respuesta)
        message = input("> ")
