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

llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=0.7,
    max_tokens=500,
    top_p=1,
    n_ctx=2048,
)
embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
client = QdrantClient(path="./local_qdrant_rubert_from_2012")
db = Qdrant(client=client, collection_name="my_news_collection", embeddings=embeddings)
retriever = db.as_retriever()

qa_sys_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Вы - чат-бот для помощи в ответах на вопросы по новостям. \
Используйте следующие фрагменты полученного контекста, чтобы ответить на вопрос. \
Если вы не знаете ответа, просто скажите, что не знаете. \
Используйте максимум три предложения и будьте краткими.
Контекст:\
{context}<|eot_id|>"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_sys_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

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
