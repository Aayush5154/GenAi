from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512,
)


model = ChatHuggingFace(llm=llm)

while True:
    user_input = input('You: ')
    if(user_input.lower() == 'exit'):
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(user_input)
    print('Ai: ', result.content)

# this is working 

# now we will add the history 
