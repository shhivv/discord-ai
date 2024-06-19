import os
import json
import chromadb.config
import ollama
import chromadb

def get_messages(directory):
    messages = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'messages.json':
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        if 'Timestamp' in item and 'Contents' in item:
                            message = f"{item['Timestamp']} - {item['Contents']}"
                            messages.append(message)
    
    return messages

if __name__ == "__main__":
    directory = 'messages'
    client = chromadb.PersistentClient(path="./mdb")
    collection = client.create_collection(name="messages")


    messages = get_messages(directory)
    print("retrived messages")

    for i, d in enumerate(messages):
      response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
      embedding = response["embedding"]
      collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
      )

