from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
) #llm congiguration for huggingface model. The repo_id parameter specifies the models repository on Hugging Face, and the task parameter indicates the type of task to perform (in this case, text generation).

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India") #invoking the model object and passing the prompt as a parameter. The invoke method sends the prompt to the Hugging Face API and returns the generated response.

print(result)