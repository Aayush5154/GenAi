#messages 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512,
)


model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content='You are a helpful AI assisteant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if(user_input.lower() == 'exit'):
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('Ai: ', result.content)

print(chat_history)    



