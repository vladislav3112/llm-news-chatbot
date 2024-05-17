FROM python:3.11

COPY . /app
WORKDIR /app
RUN mkdir -p ./models
RUN wget -nc -P ./models https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf

RUN mkdir -p ./data
RUN wget -nc -P ./data https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2
RUN apt-get update


RUN apt-get install -y bzip2 && apt -y install locales  && locale-gen ru_RU.UTF-8
RUN apt-get install -y locales-all
ENV LANG ru_RU.UTF-8
ENV LANGUAGE ru_RU:ru
ENV LC_ALL ru_RU.UTF-8

RUN bzip2 -d ./data/lenta-ru-news.csv.bz2 

RUN pip install -r requirements.txt
RUN python3 ./process_data.py
RUN rm -rf ${DF_PATH}
RUN python3 ./vectordb_save_vectors.py
RUN rm -rf ${DF_PROCESSED_PATH}
CMD [ "python3", "./run_chatbot.py"]