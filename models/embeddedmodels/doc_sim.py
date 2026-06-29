#semantic search from embeggins to find the docs similarity 
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large') #created the object of the class OpenAIEmbeddings and passed the model name as a parameter. The model_name parameter specifies the pre-trained model to use for generating embeddings.
documents = [
    "Virat kohli is a famous Indian cricketer.",
    "Sachin Tendulkar is a legendary Indian cricketer.",
    "Rohit Sharma is a talented Indian cricketer.",
    "Ms Dhoni is a former Indian cricketer and captain of the Indian national team."
]

query = "tell me about the virat kolhi" #sample query to find the most similar document from the list of documents.
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0] #isko hamesha 2d vector ki form me dena chiaye 

index, score = sorted(list(enumerate(scores)), key = lambda x : x[1])[-1]


print(query)
print(documents[index]) #printing the most similar document based on the cosine similarity score. The document with the highest score is considered the most similar to the query.
print("Similarity score is :", score)

#we will store this futhur in a vector database like weaviate or pinecone so that we can query it later on.
