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

chat_history = []

while True:
    user_input = input('You: ')
    chat_history.append(user_input)
    if(user_input.lower() == 'exit'):
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print('Ai: ', result.content)

print(chat_history)    

# now instead of mainting the list for the history which causes the problem that the 
# ye nahi batata ai ne konsa message bheja or user ne konsa its a general list 
# so to solve that dict maintain karni hogi 
# iske liye langchain me ek built in class hai jo ki chat history maintain karne ke liye use hoti hai
# that is memory - messages 
# chatbot3.py
