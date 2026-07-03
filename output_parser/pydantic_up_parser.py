from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512,
) 

model = ChatHuggingFace(llm=llm)

#pydantic object as schema for the output parser
class Person(BaseModel):
    name : str = Field(description="Name of the person")
    age : int = Field(gt= 18, description="Age of the person")
    city : str = Field(description="City of the person")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Give me the name, age and city of the fictional {place} person \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place' : 'Indian'}) # {} empty dict because we have no input variables in the prompt

print(final_result)



