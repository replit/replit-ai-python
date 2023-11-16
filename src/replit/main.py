from replit.ai.modelfarm import AsyncModelfarm, Modelfarm


async def chat():
    client = AsyncModelfarm()
    completions = await client.chat.completions.create(
        model="chat-bison@001",
        messages=[
            {
                "role": "USER",
                "content": "Hello!",
            },
        ],
        stream=True,
    )
    content = "".join([
        completion.choices[0].delta.content or ""
        async for completion in completions
    ])
    print(content)


async def completions():
    client = AsyncModelfarm()
    completion = await client.completions.create(
        model="text-bison@001",
        prompt="My name is",
    )
    print(completion.choices[0].text)


async def embeddings():
    client = AsyncModelfarm()
    response = await client.embeddings.create(model="textembedding-gecko@001",
                                              input="Hello world!")
    # Extract the AI output embedding as a list of floats
    embedding = response.data[0].embedding

    print(f"Embedding {len(embedding)}: {embedding[:5]}...")


def client_info() -> None:
    client = Modelfarm()
    print("Base url:", client.base_url)
    print("token:", client.auth.get_token())


async def main():
    #client_info()
    await chat()
    #await embeddings()
    #await completions()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
