
model = 'gpt-4-turbo-0125-spot'
azure_endpoint = "https://gcraoai9wus3spot.openai.azure.com/"


if __name__ == '__main__':
    import os
    from openai import AzureOpenAI
    api_key = os.getenv("OPENAI_API_KEY")  # need to fill this in
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_version="2024-02-01",
        api_key=api_key,
    )

    response = client.chat.completions.create(  # replace this value with the deployment name you chose when you deployed the associated model.
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the best ice cream flavor?"}
        ],
        temperature=0,
        max_tokens=350,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    print(response.choices[0].message.content)
