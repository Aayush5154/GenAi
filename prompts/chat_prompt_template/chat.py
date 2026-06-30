#these are the dynamics prompt 

from langchain_core.prompts import ChatPromptTemplate

chat_templates = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'), #this is system message and you have to pass these in tuples that is different behaviour than the normal prompt template
    ('human', 'Explain in simple terms what is {topic}') #this is human message 
])

prompt = chat_templates.invoke({'domain' : 'AI', 'topic' : 'ChatGPT'})

print(prompt)