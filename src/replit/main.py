from replit.ai.modelfarm import AsyncModelfarm, Modelfarm


async def main():
    client = AsyncModelfarm()
    completion = await client.chat.completions.create(
        model="llama2-7b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
        ],
        stream=True,
    )
    async for chunk in completion:
        print(chunk.choices[0].model_dump())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
