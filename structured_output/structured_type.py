from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

class Review(TypedDict):
    summary: str
    sentiment: str

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512,
)   

model = ChatHuggingFace(llm=llm)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The product was not good and the delivery was delayed as well as the packaging was not good and issue of size.""")

print(type(result))
print(result)
print(result['summary'])
print(result['sentiment'])

# without giving the prompt still the models itself generated the prompt 
# this was the simple TypedDict


