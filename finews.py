import os
import json
from groq import Groq
import numpy as np
from datetime import datetime
from openai import OpenAI
import random
import time
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

start_time = time.time()

encoder = SentenceTransformer("all-MiniLM-L6-v2")
print("encoder loaded")

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="finews_collections",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)

print("collection created")

with open("transformed_news_headlines.json", "r") as json_file:
    documents = json.load(json_file)

sampled_documents = random.sample(documents, 10000)

print("sample document loaded")

client.upload_points(
    collection_name="finews_collections",
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(doc["headline"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(sampled_documents)
    ],
)

print("sample document loaded to the vectorDB")

hits = client.query_points(
    collection_name="finews_collections",
    query=encoder.encode("technology new ETF").tolist(),
    limit=10,
).points

content = [hit.payload for hit in hits]
today = datetime.today()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

prompt = f''' 
Generate summarized news from user input from the context which will be given at the end of this prompt.
the context was originally json format and transformed to a list. 
The summary will focus on facts and accurate information and provide top 5 relevant news. top 5 News should be within maximum 2years from {today}. 
Provide each news output with strictly following format.
1. Date:
2. Headline : 
3. Related Stock:

context : {content}

'''

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a news intelligence agent. You are providing news related to Asset Management business. You always provide accurate and precise information in professional tone",
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
