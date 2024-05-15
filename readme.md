### Чат-бот для ответа на вопросы по новостям с применением RAG

#### Используются:
- langchain, Qdrant
- llama-cpp-python (ссылка на [модель](https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf?show_file_info=model-q4_K.gguf)) 
- [новостной датасет](https://www.kaggle.com/datasets/yutkin/corpus-of-russian-news-articles-from-lenta) 

### Как начать пользоваться:
1 вариант:
- установить зависимости из ```requirements.txt``` (например, ```pip install requrements.txt```) и задать пути в файле ```.env```
- загрузить датасет и распокавать в локальную директорию проекта, запустить ```process_data.py```
- запустить ```vectordb_save_vectors.py```, поставить флаг ```LOW_MEMORY = False``` если доступно >= 32 GB (тогда данные будут только по новостям от 2012 г.)
- запустить ```run_chatbot.py```
