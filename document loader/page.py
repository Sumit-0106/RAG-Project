from langchain_community.document_loaders import WebBaseLoader

url = "https://www.wunderground.com/history/daily/IN/Bangalore/VTBS/date/2023-09-01"
loader = WebBaseLoader([url])
data = loader.load()
print(data[0].page_content)