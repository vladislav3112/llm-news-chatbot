from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
from tqdm import tqdm

# setting if device free RAM <= 16 GB (but >= 8GB)
LOW_MEMORY = True

if LOW_MEMORY:
    df = pd.read_csv("data/processed_news.csv").sample(frac=0.2, random_state=0)
else:
    df = pd.read_csv("data/processed_news.csv")

documents = []

for item in tqdm(df.itertuples(), total=len(df)):
    page = Document(
        page_content=item.text, metadata={"date": item.date, "topic": item.topic}
    )
    documents.append(page)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)


# The embedding model that will be used by the collection
embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")

vectorstore = Qdrant.from_documents(
    documents=splits,
    embedding=embeddings,  # LlamaCppEmbeddings(model_path=LLM_MODEL_PATH),
    path="/local_qdrant_rubert",
    collection_name="my_news_collection",
)
