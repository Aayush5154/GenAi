from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

load_dotenv()

def word_count(input):
    return len(input.split())


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

prompt1 = PromptTemplate(
    template = 'Write a joke about the topic : {topic}',
    input_variables = ['topic']
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' :  RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count)
    # another way is 'word_count' : RunnableLambda(lambda x : len(x.split()))    
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "programming"})

final_result = """{} \n word Count - {}""".format(result['joke'], result['word_count'])

print(final_result)



