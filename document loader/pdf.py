from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("document loader/GRU.pdf")
docs = data.load()

splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=1)

chunks = splitter.split_documents(data.load())

print(len(chunks))