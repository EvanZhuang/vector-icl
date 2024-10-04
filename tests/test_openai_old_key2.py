import os
import openai
from openai import AzureOpenAI


if __name__ == '__main__':
    api_key = os.getenv("OPENAI_API_KEY_NEW")  # need to fill this in
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint="https://gcrendpoint.azurewebsites.net/",
    )

    response = client.chat.completions.create(  # replace this value with the deployment name you chose when you deployed the associated model.
        # model='gpt-4',
        model='gpt-4-turbo',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "2+3="}
        ],
        temperature=0,
        # max_tokens=3,
        # top_p=0.95,
        # frequency_penalty=0,
        # presence_penalty=0,
        # stop=None
    )
    print(response.choices[0].message.content)
