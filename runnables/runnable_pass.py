# this is runnable passthrough
# joki input hi o/p me deta hain 

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'Write a joke about the topic : {topic}',
    input_variables = ['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke in simple terms: {joke}',
    input_variables=['joke']
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)
 
parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'exlanation' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({"topic": "programming"}))

print(final_chain.invoke({"topic" : "programming"})['joke']) # joke is passed to the explanation chain as input