import pandas as pd
from dotenv import load_dotenv
import os


def process_news_df(path: str, save_path: str) -> None:
    """Функция для обработки датафрейма с данными

    Args:
        path (str): путь к датасету
        path (str): путь, куда сохратить обработанный датасет
    """
    # ## Load and processing
    df_news = pd.read_csv(path)
    df_news["date"] = pd.to_datetime(df_news["date"])
    MIN_COUNT = len(df_news) * 0.005

    # ### Select only if text exists:
    df_news = df_news[df_news["text"].notna()]

    # ### Null percentages and topic filtration:
    topics_to_use = df_news.topic.value_counts()[
        df_news.topic.value_counts() > MIN_COUNT
    ].index
    df_news = df_news[df_news.topic.isin(topics_to_use)]
    df_news = df_news[~df_news["text"].str.contains("\(function")]
    df_news[["title", "text", "topic", "date"]].to_csv(save_path)


if __name__ == "__main__":

    load_dotenv()
    DF_PATH = os.getenv("DF_PATH")
    DF_PROCESSED_PATH = os.getenv("DF_PROCESSED_PATH")
    process_news_df(path=DF_PATH, save_path=DF_PROCESSED_PATH)
