from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Qdrant
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from qdrant_client import QdrantClient
import langchain

langchain.verbose = True
langchain.debug = True

# load vectors + LLM:
# model source: https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf


LLM_MODEL_PATH = "G:/LLMs weights/model-q4_K.gguf"
DB_PATH = "./local_qdrant_rubert_from_2012"
store = {}


def create_retriever(db_path: str, embeddings: HuggingFaceEmbeddings):
    """Создание объекта retriever из локально сохраненной базы данных

    Args:
        db_path (str): путь к локально сохраненной векторной БД
        embeddings (HuggingFaceEmbeddings): модель для извлечения векторных представлений

    Returns:
        _type_: _description_
    """
    client = QdrantClient(path=db_path)
    db = Qdrant(
        client=client, collection_name="my_news_collection", embeddings=embeddings
    )
    retriever = db.as_retriever()
    return retriever


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def create_rag_chain(
    llm: LlamaCpp, qa_sys_prompt: str, db_retriever
) -> RunnableWithMessageHistory:
    """Функция для создания RAG chain

    Args:
        llm (LlamaCpp): LLM модель, что будет использваться для получения отвеета
        qa_sys_prompt (str): системное сообщение
        db_retriever (_type_): объект retriever БД, можно получить вызовом create_retriever

    Returns:
        RunnableWithMessageHistory: conversational_rag_chain с которым и будет происодить основное взаимодействие
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_sys_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(db_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def run_chat_boot(llm: LlamaCpp, sys_prompt: str, db_retriever) -> None:
    global store
    store = {}

    conversational_rag_chain = create_rag_chain(llm, sys_prompt, db_retriever)
    while True:
        print("Введите вопрос. Напишите quit или q чтобы выйти, r для перегенерации.")

        user_message = input()
        if user_message == "q" or user_message == "quit":
            break
        elif user_message == "r":
            store = {}
            print("Сброс произошёл успешно!")
        else:
            print(
                "Ответ:",
                conversational_rag_chain.invoke(
                    {
                        "input": f"<|start_header_id|>user<|end_header_id|>{user_message}<|eot_id|>"
                    },
                    config={
                        "configurable": {"session_id": "abc123"}
                    },  # constructs a key "abc123" in `store`.
                )["answer"],
            )


if __name__ == "__main__":

    db_retriever = create_retriever(
        db_path="./local_qdrant_rubert_from_2012",
        embeddings=HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2"),
    )

    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        n_ctx=2048,
    )

    sys_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    Вы - чат-бот для помощи в ответах на вопросы по новостям. \
    Используйте следующие фрагменты полученного контекста, чтобы ответить на вопрос. \
    Если вы не знаете ответа, просто скажите, что не знаете. \
    Используйте максимум три предложения и будьте краткими.
    Контекст:\
    {context}<|eot_id|>"""

    run_chat_boot(llm, sys_prompt, db_retriever)
