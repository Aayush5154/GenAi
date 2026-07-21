from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('agri-sure.pdf')

docs = loader.load()

print(len(docs))

print(docs[3].page_content)
print(docs[3].metadata)

#  pypdf loader is mainly used for the textual data 
#  for the table -> pdfLumberLoader
#  scnned pdf image -> unstructuredPdfLoader

