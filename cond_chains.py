# feedback system 
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Sentiment(BaseModel):
    #sentiment is the attribute 
    sentiment : Literal['positive', 'negative'] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Sentiment)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback as positive, negative \n: "{feedback}" \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

result = classifier_chain.invoke({"feedback": "I love this product! It works teribale."}).sentiment # but this ouptput is not structured, we want to use pydantic output parser to structure the output 
#may give the output like this feedback is positive/negative but we need in single word like postive/negative so we are usign teh pydantic output parser to structure the output


print(result) # Output: positive

#section 2 runnable branch where we send multiple tuples 

prompt2 = PromptTemplate(
    template='Generate a positive response to the following feedback: "{feedback}"',
    input_variables=['feedback']
)

# classifier_chain only passes Sentiment(sentiment="positive") to branch_chain, not the original {"feedback": ...}, so prompt2 cannot fill the {feedback} placeholder unless the feedback is explicitly passed along.

prompt3 = PromptTemplate(
    template='Generate a negative response to the following feedback: "{feedback}"',
    input_variables=['feedback']
)

# branch_chain = RunnableBranch(
#     (condition1, chain1), # if
#     (condition2, chain2), # else if
#     # else case
# )

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not find sentiment")
) 


chain = classifier_chain | branch_chain

result2 = chain.invoke({"feedback": "I love this product! It works good."})

print(result2)


