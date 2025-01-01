import os
import json
import numpy as np
import openai
import random
import time

start_time = time.time()
 
openai.api_key = os.environ["OPENAI_API_KEY"]
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

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
    query=encoder.encode("ETF new launch").tolist(),
    limit=3,
).points

for hit in hits:
    print(hit.payload['headline'], "score:", hit.score)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
