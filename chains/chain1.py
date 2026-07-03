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


prompt = PromptTemplate(
    template='Generate 5 interesting facts about the following topic: {topic}',
    input_variables=['topic']
)


model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser #pipe operator (|) is used to chain the prompt, model and parser together. The output of the prompt will be passed as input to the model and the output of the model will be passed as input to the parser.

result = chain.invoke({'topic': 'Cricket'})

# print(result)

chain.get_graph().print_ascii() 


