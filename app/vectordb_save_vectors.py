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
    take_sample,
    low_memory=True,
):
    """Сохраняет данные в векторную базу данных.

    Args:
        df_path (str, optional): _description_. Путь к данным после запуска чкрипта обработки данных.
        db_save_path (str, optional): _description_. Путь, куда соханятся данные векторной базы данных.
        take_sample (bool, optional): _description_. Defaults to True. Флаг: брать случайную быборку из 1000 новостей
                    после 2012 г. или же все новости. Использовать исключительно в тестовых целях.
                    При нехватке RAM лучше использовать low_memory=True
        low_memory (bool, optional): _description_. Defaults to True.
                    Флаг: оставить только новости после 2012 г или нет.

    """

    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"])
    if take_sample:

        df = df[df.date.dt.year >= 2012].sample(1000, random_state=0)
    elif low_memory:
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
    TAKE_SAMPLE = os.getenv("TAKE_SAMPLE")
    LOW_MEMORY = os.getenv("LOW_MEMORY")
    save_data_to_vectordb(
        df_path=DF_PROCESSED_PATH,
        db_save_path=DB_PATH,
        take_sample=TAKE_SAMPLE,
        low_memory=LOW_MEMORY,
    )
