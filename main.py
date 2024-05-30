import os
import getpass
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_APIKEY")


# step 1: load the document
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/"), 
    bs_kwargs={"parse_only": bs4_strainer}
    )

docs = loader.load()

# step 2: split the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

# step 3: Indexing 
vector_store = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# step 4: retrieve the documents
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke(input="What are the approaches to task decomposition?")

print(len(retrieved_docs))

# step 5: chat with the retrieved documents
llm = ChatOpenAI()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    |llm
    | StrOutputParser()
)

answer = chain.invoke(input="What are the approaches to task decomposition?")

print(answer)



