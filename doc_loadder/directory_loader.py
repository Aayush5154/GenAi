# directry loader is a documnet loader that let u load multiple documents from a directory of files.
# .txt, .pdf, .docx, .csv, .json, .html, .md, .pptx, .epub, .xml, .yaml, .yml, .tsv, .rst, .odt

from langchain_community.document_loaders import DirectoryLoader, pyPDFLoader

loader = DirectoryLoader(
    path = 'books',
    glob = '*.pdf',
    loader_cls = pyPDFLoader
)
# from the books folder extracts all the pdf files and loads them into a list of documents
# and all the pages will be asasigned from the index 0.

docs = loader.load()

print(len(docs))

print(len(docs[0].page_content))
print(docs[0].metadata)

# the probblme in this is that it becomes very difficult to load more than 5 pdfs at the same times 
# because u are laoding them into the ram 
# so the solution is -> lazy loading 

# ----------->

# load() -> for the number of dics less then 5 when u all want them at the same time 

# lazy - loading -> for the number of docs more than 5 when u want to load them one by one and not all at the same time.
# only when they required.

# ham ek load karte h then uska meta data lete h and then hata dete hain then again will take the new one.

