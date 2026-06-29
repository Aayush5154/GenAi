from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Free Hugging Face API
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Virat Kohli is a famous Indian cricketer.",
    "Sachin Tendulkar is a legendary Indian cricketer.",
    "Rohit Sharma is a talented Indian cricketer.",
    "MS Dhoni is a former Indian cricketer and captain of the Indian national team."
]

query = "Tell me about Rohit Sharma"

# Generate embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

best_index = scores.argmax()

print("Query:", query)
print("Most Similar Document:")
print(documents[best_index])
print("Similarity Score:", scores[best_index])