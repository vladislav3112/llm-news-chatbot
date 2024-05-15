import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from dotenv import load_dotenv
import os


def save_data_to_vectordb(
    df_path,
    db_save_path,
    low_memory=True,
):
    """Сохраняет данные в векторную базу данных.

    Args:
        df_path (str, optional): _description_. Путь к данным после запуска чкрипта обработки данных.
        db_save_path (str, optional): _description_. Путь, куда соханятся данные векторной базы данных.
        low_memory (bool, optional): _description_. Defaults to True. Флаг, >= 16GB RAM на устройстуе или нет.
    """

    df = pd.read_csv(df_path)
    if low_memory:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df.date.dt.year >= 2012]

    documents = []

    for item in tqdm(df.itertuples(), total=len(df)):
        page = Document(
            page_content=f"Дата новости:{item.date} {item.text}",
        )
        documents.append(page)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # The embedding model that will be used by the collection
    embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")

    try:
        Qdrant.from_documents(
            documents=splits,
            embedding=embeddings,  # LlamaCppEmbeddings(model_path=LLM_MODEL_PATH),
            path=db_save_path,
            collection_name="my_news_collection",
        )
        print(f"Информация успешно сохранена в базу данных по пути {db_save_path}")
    except MemoryError:
        print("Недостаточно памяти! Освободите RAM или поставьте low_memory=True")


if __name__ == "__main__":

    load_dotenv()

    DB_PATH = os.getenv("DB_PATH")
    DF_PROCESSED_PATH = os.getenv("DF_PROCESSED_PATH")
    save_data_to_vectordb(
        df_path=DF_PROCESSED_PATH,
        db_save_path=DB_PATH,
    )
