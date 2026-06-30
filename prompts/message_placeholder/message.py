from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer supoort agent'),
    MessagesPlaceholder(variable_name='chat_history'), #pich up the chathistory and every tim before the query takes here for ref of old messages 
    ('human', '{query}')
])

chat_history = []

#load chat history

with open('chat_history.txt') as f :
    chat_history.extend(f.readlines())

print(chat_history)

#create prompt 

prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'Where is my refund'})


print(prompt)

#every times it fetches fromt the previous data before the query and then it will generate the prompt for the new query