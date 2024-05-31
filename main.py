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


# todo: обновить
def load_dataset( filename:str ):
    """Загружает датасет из вопросов и ответов.
    Формат файла:
    Вопрос? <одна строка>
    Ответ,
    ответ может занимает несколько абзацев или строк
    @return: DataFrame(columns=['Q', 'A'])"""

    Q = []      # вопросы
    A = []      # ответы
    print('Начало обработки файла')
    # загрузка вопросов и ответов из файла
    with open( filename ) as f_in:
        json_dict = json.load(f_in)
        print(type(json_dict))
    print('Конец обработки файла')
    data = pd.DataFrame( {"Q":Q, "A":A})
    return data


# класс-обёртка для создания эмбеддингов текстов
from langchain_community.embeddings import HuggingFaceEmbeddings

# сравнительно простая (и быстрая) модель, выдаёт эмбеддинги () для текстов

model_name = "cointegrated/LaBSE-en-ru"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings_maker = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


data = load_dataset("data/dataset.json")
loader = DataFrameLoader(data, page_content_column='Q')   
documents = loader.load()

# объект, который будет разбивать тексты из датасета на блоки, если вдруг они будут слишком большими для модели выдающий эмбеддинги
text_splitter = RecursiveCharacterTextSplitter(chunk_size = embeddings_maker.client.max_seq_length, chunk_overlap=0)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 50, chunk_overlap=0)        # для примера
# chunk_size - это размер блока в токенах, будет разбивать на части только ключ (здесь, это вопрос)

# получим блоки. Блок = (вопрос (ключ), ответ);
# Вопрос может быть не полным, если не поместится в chunk_size. Тогда создаётся новый блок, с остатком вопроса, но с таким же ответом.
texts = text_splitter.split_documents(documents)
print(f"текстов: {len(texts)}")          # при максимальном размере вопроса в токенах 384, разбивать вопросы на части не пришлось.
# texts

      # зададим ключ для поиска по текстам, это колонка с вопросом\n",


# # создаем хранилище
# db = FAISS.from_documents(texts, embeddings_maker)
# db.as_retriever()           # ???

# # пример использования:
# db.similarity_search_with_score('Как участвовать на нескольких хакатонах?', k = 5 )
# # поданный запрос переводится в эмбеддинг, для него выдаётся топ K самых похожих частей датасета (вопрос, ответ, расстояние)


# import ollama

# # request = 'Мне 10 лет. Я могу участвовать в хакатоне?'
# # request = 'Мне 100 лет. Я могу участвовать в хакатоне?'
# # request = 'Сколько участников должно быть в команде?'
# # request = 'Как участвовать на нескольких хакатонах?'


# request = input("Вопрос: ")

# context = db.similarity_search_with_score(request, k = 5 )
# context = " ".join([text[0].metadata['A'] for text in context])
# print(context)

# response = ollama.chat(model='dimweb/ilyagusev-saiga_llama3_8b:Q6_K', messages=[
#   {
#     'role': 'user',
#     'content': f'Дай развёрнутый и как можно более точный ответ. Для ответа используй дополнительную информацию.\nВопрос: {request}.\n Дополнительная информация: {context}',}],
#     stream = True
# )
# # print(response['message']['content'])

# for chunk in response:
#   print(chunk['message']['content'], end='', flush=True)