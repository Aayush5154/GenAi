# we can inforce the structure but still there is not valiadation 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   
from langchain.output_parsers.structured import (StructuredOutputParser, ResponseSchema)
#this has been removed from the latest version of langchain_core.output_parsers.structured 
#so this will not work 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512,
) 

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact_1', description='First fact about the topic'),
    ResponseSchema(name='fact_2', description='Second fact about the topic'),
    ResponseSchema(name='fact_3', description='Third fact about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about the {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'topic' : 'black hole'}) 

print(final_result)

