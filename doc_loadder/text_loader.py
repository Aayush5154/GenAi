from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
)

prompt = PromptTemplate(
    template = 'Write a summary of the following poem:\n{text}\n\nSummary:',
    input_variables = ['text']
)

parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0]) # this will contain the page_content and meta data 


chain = prompt | model | parser

print("Here is the summary of the poem:")

print(chain.invoke({'text': docs[0].page_content}))




