import os
import openai
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(azure_credential,
                                           "https://cognitiveservices.azure.com/.default")


if __name__ == '__main__':
    client = AzureOpenAI(
        api_version="2024-02-15-preview",
        azure_endpoint="https://dl-openai-1.openai.azure.com/",
        azure_ad_token_provider=token_provider,
    )

    response = client.chat.completions.create(  # replace this value with the deployment name you chose when you deployed the associated model.
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "2+3="}
        ],
        temperature=0,
    )
    print(response.choices[0].message.content)
