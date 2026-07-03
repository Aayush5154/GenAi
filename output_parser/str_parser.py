from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512,
) 

model = ChatHuggingFace(llm=llm)

# 1st prompt - > detailed report 

template1 = PromptTemplate(
    template='Write a detailed report on the following topic: {topic}',
    input_variables=['topic'] 
)

# 2nd prompt -> summarize the report in 3 sentences

template2 = PromptTemplate(
    template='Summarize the 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic : blackhole'})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})

result2 = model.invoke(prompt2)

print("Detailed Report:\n", result2.content)