import discord
import ollama
import chromadb
import asyncio
from datetime import datetime
from typing import List


class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        self.chroma = chromadb.PersistentClient(path="./mdb")

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.content.startswith("!"):
            content = message.content[1:]
            await self.handle_command(message, content)

    async def handle_command(self, message, content):
        collection = self.chroma.get_collection("messages")

        response = ollama.embeddings(
            prompt=content,
            model="mxbai-embed-large"
        )
        results = await self.query_collection(collection, response["embedding"])

        sorted_messages = self.sort_messages(results['documents'][0])
        output = await self.generate_response(content, sorted_messages)

        await self.send_response(message, output)

    async def query_collection(self, collection, embedding):
        return await asyncio.to_thread(
            lambda: collection.query(
                query_embeddings=[embedding],
                n_results=20
            )
        )

    def sort_messages(self, messages: List[str]) -> List[str]:
        parsed_messages = [
            (datetime.strptime(message.split(" - ")[0], "%Y-%m-%d %H:%M:%S"), message)
            for message in messages
        ]
        return [message[1] for message in sorted(parsed_messages, key=lambda x: x[0])]

    async def generate_response(self, prompt, messages):
        return await asyncio.to_thread(
            lambda: ollama.generate(
                model="phi3",
                prompt=f"There are my messages from over the past few years, represented by timestamp followed by content. Using this information: {messages}. Respond to this prompt by providing a summary in a concise manner: {prompt}"
            )
        )

    async def send_response(self, message, output):
        chunks = self.chunk_string(output['response'])
        for chunk in chunks:
            await message.reply(chunk)
            await asyncio.sleep(1)

    @staticmethod
    def chunk_string(input_string, chunk_size=2000):
        if len(input_string) <= chunk_size:
            return [input_string]

        if chunk_size <= 0:
            raise ValueError("Chunk size must be a positive integer.")

        return [input_string[i : i + chunk_size] for i in range(0, len(input_string), chunk_size)]


intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
client.run('--Add your Discord token here--')
