# we will we creating the parallel chians 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,   
    max_new_tokens=512,
)

# llm2 = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#     task="text-generation",
# )

prompt1 = PromptTemplate(
    template='Generate short and simple text from the following text : {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate a 10 short question answers from teh following text : {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single ddocument: {notes} {quiz}',
    input_variables=['notes', 'quiz']
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({'text' : 'The Industrial Revolution was a period of significant technological advancements that transformed the way goods were produced and consumed. It began in the late 18th century and continued into the 19th century, primarily in Britain, before spreading to other parts of the world. The revolution was characterized by the introduction of machinery, the development of new manufacturing processes, and the growth of industries such as textiles, iron, and coal mining. It led to urbanization, changes in labor practices, and had profound social and economic impacts on society.'})

chain.get_graph().print_ascii()
# print(result) 

