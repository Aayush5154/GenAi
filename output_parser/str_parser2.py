from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
# so here the parser comes which connects the output of the first model to the input of the second model
# else aapko do chains banani padti 

result = chain.invoke({'topic': 'blackhole'})

print(result)



