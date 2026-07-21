from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)   

model = ChatHuggingFace(llm=llm)   

prompt1 = PromptTemplate(
    template = 'Generate a tweet about the topic : {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a LinkedIn post about the topic : {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedin_post' : RunnableSequence(prompt2, model, parser)
}) # A dict will be made with first key is tweet and second key will be linkdin_post

result = parallel_chain.invoke({"topic": "Ai"})

print(result['tweet'])

