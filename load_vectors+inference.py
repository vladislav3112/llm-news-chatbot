from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

template_messages = [
    SystemMessage(content="You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{text}"),
]
prompt_template = ChatPromptTemplate.from_messages(template_messages)

# model source: https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf
LLM_MODEL_PATH = "G:/LLMs weights/model-q4_K.gguf"

embeddings = HuggingFaceEmbeddings(model_name="cointegrated/rubert-tiny2")
client = QdrantClient(path="/local_qdrant_rubert")
db = Qdrant(client=client, collection_name="my_news_collection", embeddings=embeddings)
retriever = db.as_retriever()


template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Вы помощник в ответах на вопросы по новостям.
Используйте следующие фрагменты полученного контекста, чтобы ответить на вопрос.
Если вы не знаете ответа, просто скажите, что не знаете.
Используйте максимум три предложения и будьте краткими.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Вопрос: {question} 
Контекст: {context}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Ответ:
"""

prompt = PromptTemplate.from_template(template)
# print(prompt)
# prompt.messages

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    temperature=0.7,
    max_tokens=500,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    n_ctx=1024,
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("Выедите вопрос.")
rag_chain.invoke(input())
