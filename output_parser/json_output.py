from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=512,
) 

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template1 = PromptTemplate(
    # template='Give me the name, age and city of the fictional person \n {format_instructions}',
    template = 'Give me 5 facts about {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# prompt = template1.format()

# # print("Prompt:\n", prompt)

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

# you can also use the chain to connect the prompt, model and parser together

chain = template1 | model | parser

final_result = chain.invoke({'topic' : 'black hole'}) # {} empty dict because we have no input variables in the prompt


# print(final_result['name']) # its making the dict 
print(final_result)
print(type(final_result))


