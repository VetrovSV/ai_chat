"""
Главный файл приложения
"""

import pandas as pd
# from langchain.document_loaders import DataFrameLoader                  # отдельный тип для хранения датафреймов (depricated)
from langchain_community.document_loaders import DataFrameLoader

# будет разбивать тексты на части, чтобы делать их них эмбеддинги
from langchain.text_splitter import RecursiveCharacterTextSplitter      

# todo: разобраться с предупреждениями

# класс для хранения данных как в векторной БД?. Используется для быстрого поиска подходящего контекста по запросу
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

# загрузка датасета, из которого будет использоваться информация для дополнения промпта
import json

# класс-обёртка для создания эмбеддингов текстов
from langchain_community.embeddings import HuggingFaceEmbeddings


# название модели для получения эмебддингов
EMB_MODEL_NAME = "cointegrated/LaBSE-en-ru"
# название большой языковойй модели
LLM_NAME = "dimweb/ilyagusev-saiga_llama3_8b:Q6_K"
LLM_NAME = "gemma:2b"

# загрузка модели эмбеддингов
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings_maker = HuggingFaceEmbeddings(
    model_name=EMB_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# загрузка датасета
data = pd.json_normalize( pd.read_json("data/dataset.json")['data'])
# json_normalize, чтобы избавиться от корневого элемента, сделать плоский датафреим
print(f"Загружено документов: {len(data)}" )


# создание БД из датасета
loader = DataFrameLoader(data, page_content_column='title')
documents = loader.load()
# объект, который будет разбивать тексты из датасета на блоки, если вдруг они будут слишком большими для модели выдающий эмбеддинги
text_splitter = RecursiveCharacterTextSplitter(chunk_size = embeddings_maker.client.max_seq_length, chunk_overlap=0)
# chunk_size - это размер блока в токенах, будет разбивать на части только ключ (здесь, это вопрос)
# получим блоки. Блок = (вопрос (ключ), ответ);
# Вопрос может быть не полным, если не поместится в chunk_size. Тогда создаётся новый блок, с остатком вопроса, но с таким же ответом.
texts = text_splitter.split_documents(documents)
print(f"текстов: {len(texts)}")          # при максимальном размере вопроса в токенах 384, разбивать вопросы на части не пришлось.

# создаем хранилище
print("создаем хранилище... ", end="")
db = FAISS.from_documents(texts, embeddings_maker)
print("Создано")
# db.as_retriever()           # ???

# todo: тут нужно сохранить БД в отдельное место. Иначе создание занимает пару минут
# # пример использования:

# print(
# db.similarity_search_with_score('Как перевыпустит карту', k = 5 )
# )
# # поданный запрос переводится в эмбеддинг, для него выдаётся топ K самых похожих частей датасета (вопрос, ответ, расстояние)


import ollama

# # request = 'Мне 10 лет. Я могу участвовать в хакатоне?'
# # request = 'Мне 100 лет. Я могу участвовать в хакатоне?'
# # request = 'Сколько участников должно быть в команде?'
# # request = 'Как участвовать на нескольких хакатонах?'


request = input("Вопрос: ")

print("\nКонтекст:")
context = db.similarity_search_with_score(request, k = 5 )
context = " ".join([text[0].metadata['description'] for text in context])
print(context)

response = ollama.chat(model=LLM_NAME, messages=[
  {
    'role': 'user',
    'content': f'Дай развёрнутый и как можно более точный ответ. Для ответа используй дополнительную информацию.\nВопрос: {request}.\n Дополнительная информация: {context}',}],
    stream = True
)


print("\n\n Ответ:")
for chunk in response:
  print(chunk['message']['content'], end='', flush=True)