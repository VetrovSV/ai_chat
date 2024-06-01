# Способ1: оценка качеста на сравнении 3 ответов

#!pip install langchain_community

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
# будет разбивать тексты на части, чтобы делать их них эмбеддинги
from langchain.text_splitter import RecursiveCharacterTextSplitter

# класс-обёртка для создания эмбеддингов текстов
from langchain_community.embeddings import HuggingFaceEmbeddings

# Класс для хранения данных как в векторной БД?. Используется для быстрого поиска подходящего контекста по запросу
from langchain_community.vectorstores import FAISS

# название модели для получения эмебддингов
EMB_MODEL_NAME = "cointegrated/LaBSE-en-ru"

def init_emb_model(model_name:str, model_kwargs:dict, encode_kwargs:dict):
    """Скачивает (если нужно) языковую модель для эмбеддингов, возвращает её"""
    print("Создание\скачивание модели эмбеддингов текстов... ", end="")
    # загрузка модели эмбеддингов
    # model_kwargs = {'device': 'cpu'}
    # encode_kwargs = {'normalize_embeddings': False}
    # сделать более точную настройку параметров, если необходимо
    Embeddings_maker = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # todo: копировать уже скаченную модель в докер
    print("Готово")
    return Embeddings_maker


def load_dataset(filename_json: str, embeddings_maker):
    """Загружает датасет, создаёт векторную БД
    @param filename_json -- JSON файл с датасетом вопросов и ответов
    @param embeddings_maker -- языковая модель для создания эмбеддингов
    @return тексты в формате пакета langchain_community
     """
    # загрузка датасета
    print("Загрузка датасета")
    data = pd.json_normalize(pd.read_json(filename_json)['data'])
    # json_normalize, чтобы избавиться от корневого элемента, сделать плоский датафреим
    print(f"Загружено документов: {len(data)}")

    # создание БД из датасета
    loader = DataFrameLoader(data, page_content_column='title')     # title -  вопрос
    documents = loader.load()
    # объект, который будет разбивать тексты из датасета на блоки, если вдруг они будут слишком большими для модели выдающий эмбеддинги
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=embeddings_maker.client.max_seq_length, chunk_overlap=0)
    # chunk_size - это размер блока в токенах, будет разбивать на части только ключ (здесь, это вопрос)
    # получим блоки. Блок = (вопрос (ключ), ответ);
    # Вопрос может быть не полным, если не поместится в chunk_size. Тогда создаётся новый блок, с остатком вопроса, но с таким же ответом.
    texts = text_splitter.split_documents(documents)
    print(
        f"текстов: {len(texts)}")  # при максимальном размере вопроса в токенах 384, разбивать вопросы на части не пришлось.
    return texts



  Embeddings_maker = init_emb_model(model_name=EMB_MODEL_NAME,
                                               model_kwargs={'device': 'cpu'},
                                               encode_kwargs={'normalize_embeddings': False})

  ###------------------>
import requests
import json

# URL API
API_URL = "http://95.189.96.144:60003/assist"


def test_api(test_query):
    # Тестовый запрос
    test_query = test_query

    # Формирование данных запроса
    request_data = {
        "query": test_query
    }


    response = requests.post(API_URL, json=request_data)

    response_data = response.json()
    response_data['text']

  # Получение текста ответа
    answer_text = response_data['text']

    return answer_text

# Вызов функции для тестирования API
answ=test_api("как открыть кредит?")
answ2=test_api("как открыть кредит?")
answ3=test_api("как открыть кредит?")


# работа с векторами
import numpy as np
from scipy.spatial.distance import cosine, euclidean

# векторы
answer1_vector = Embeddings_maker.embed_query(answ)
answer2_vector = Embeddings_maker.embed_query(answ2)
answer3_vector = Embeddings_maker.embed_query(answ3)

# косин сходство
cosine_similarity_12 = 1 - cosine(answer1_vector, answer2_vector)
cosine_similarity_13 = 1 - cosine(answer1_vector, answer3_vector)
cosine_similarity_23 = 1 - cosine(answer2_vector, answer3_vector)

print("Косинусное сходство:")
print(f"Ответ 1 и Ответ 2: {cosine_similarity_12:.2f}")
print(f"Ответ 1 и Ответ 3: {cosine_similarity_13:.2f}")
print(f"Ответ 2 и Ответ 3: {cosine_similarity_23:.2f}")

# евклидовое расстояния
euclidean_distance_12 = euclidean(answer1_vector, answer2_vector)
euclidean_distance_13 = euclidean(answer1_vector, answer3_vector)
euclidean_distance_23 = euclidean(answer2_vector, answer3_vector)

print("\nЕвклидово расстояние:")
print(f"Ответ 1 и Ответ 2: {euclidean_distance_12:.2f}")
print(f"Ответ 1 и Ответ 3: {euclidean_distance_13:.2f}")
print(f"Ответ 2 и Ответ 3: {euclidean_distance_23:.2f}")


#Способ2: Сравнение ответа бота с ответом в датасете 

DB_FAISS = "dataset.faiss"

DB = FAISS.load_local(folder_path=DB_FAISS, embeddings=Embeddings_maker, allow_dangerous_deserialization=True)

import requests
import json

# URL API
API_URL = "http://95.189.96.144:60003/assist"


def test_api(test_query):
    # Тестовый запрос
    test_query = test_query

    # Формирование данных запроса
    request_data = {
        "query": test_query
    }

    response = requests.post(API_URL, json=request_data)

    response_data = response.json()
    response_data['text']

    # Получение текста ответа
    answer_text = response_data['text']

    return answer_text


# Вызов функции для тестирования API
answ = test_api("как открыть кредит?")
ques_vector = Embeddings_maker.embed_query("как открыть кредит?")

answer_vector = Embeddings_maker.embed_query(answ)

 context = DB.similarity_search_with_score("как открыть кредит?", k=1)
    # print(context)
    # идекс 0 - документ
    # # идекс 1 - похожесть
context = "\n\n".join([ "Вопрос: " + text[0].page_content +
                       "\nОтвет: " +  text[0].metadata['description'] +
                            "\nСсылка: " + text[0].metadata['url'] for text in context])

context = DB.similarity_search_with_score("как открыть кредит?", k=1)
if context:
    answer = context[0][0].metadata['description']
    print(answer)

context_vect=Embeddings_maker.embed_query(answer)

import numpy as np
from scipy.spatial.distance import cosine, euclidean

# косин сходство
cosine_similarity = 1 - cosine(context_vect,answer_vector)


print("Косинусное сходство:")
print(f"Ответ: {cosine_similarity:.2f}")


# евклидовое расстояния
euclidean_distance= euclidean(context_vect, answer_vector)

print("\nЕвклидово расстояние:")
print(f"Ответ: {euclidean_distance:.2f}")

