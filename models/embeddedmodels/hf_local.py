from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #created the object of the class HuggingFaceEmbeddings and passed the model name as a parameter. The model_name parameter specifies the pre-trained model to use for generating embeddings.

text = "This is a sample text for generating embeddings." #sample text to generate embeddings for.

vector = embeddings.embed_query(text) #invoking the embed_query method of the embeddings object and passing the sample text as a parameter. The embed_query method generates a vector representation (embedding) of the input text.

print(str(vector)) #printing the generated vector representation of the input text. The str() function is used to convert the vector to a string format for display.

# but this requires the pytorch library to be installed in the environment. If pytorch is not installed, the code will raise an ImportError indicating that the required library is missing.

# locally download karna ho to

