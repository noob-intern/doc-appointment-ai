import os
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv('MISTRAL_API_KEY')
model = "mistral-small-latest"


client = Mistral(api_key=api_key)


chat_response = client.chat.complete(
    model= model,
    messages = [
        {
            "role": "user",
            "content": "What is the best French bread?",
        },
    ]
)
print(chat_response.choices[0].message.content)