# webpage_load ->  use to load and extracts the content from the webpages (urls)
# it uses the beautifulSoup under the hood to parse the html and extract the visible content.

# loads only the static html before the rendering.

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.7,
)

prompt = PromptTemplate(
    template = 'Answer the following \n {question} from the following context:\n {context}\n\nAnswer:',
    input_variables = ['question','text']
)

url = 'https://www.flipkart.com/apple-macbook-pro-m3-8-gb-1-tb-ssd-macos-sonoma-mtl83hn-a/p/itmcd041a34ee857?pid=COMGUTX7ENVNSZGS&lid=LSTCOMGUTX7ENVNSZGSIOLIZK&marketplace=FLIPKART&store=6bo%2Fb5g&srno=b_1_1&otracker=browse&fm=organic&iid=9de86cf4-63fc-42d8-b017-48ede6eaa879.COMGUTX7ENVNSZGS.SEARCH&ppt=None&ppn=None&ssid=kek3dl51g00000001784539539316&ov_redirect=true'

loader = WebBaseLoader(url)

docs = loader.load()

parser = StrOutputParser()

model = ChatHuggingFace(llm=llm)

# docs = loader.load()

chain = prompt | model | parser

# print("Here is the summary of the poem:")

print(chain.invoke({'question':'What is the product is about ', 'text': docs[0].page_content}))


