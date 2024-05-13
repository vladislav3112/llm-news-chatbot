import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# setting if device free RAM ~ 16 GB
LOW_MEMORY = True
df = pd.read_csv("data/processed_news.csv")
if LOW_MEMORY:
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date.dt.year >= 2012]

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
    path="./local_qdrant_rubert_from_2012",
    collection_name="my_news_collection",
)
