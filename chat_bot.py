"""
Тут основные константы и функции для работы RAG: соезинение с LLM, эмбеддинги текстов, поиск по БД и т.д.
"""

import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
# будет разбивать тексты на части, чтобы делать их них эмбеддинги
from langchain.text_splitter import RecursiveCharacterTextSplitter
# класс-обёртка для создания эмбеддингов текстов
from langchain_community.embeddings import HuggingFaceEmbeddings
# Класс для хранения данных как в векторной БД. Используется для быстрого поиска подходящего контекста по запросу
from langchain_community.vectorstores import FAISS
# Для отправки GET и POST запроса. Нужен для отправки запроса к YandexGPT Pro
import requests
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# отключить предупреждения об изменении API загрузки модели эмбеддингов

# номер телефона поддержки
SUPPORT_PHONE = "8 800 3 1415"
MAIN_HELP_PAGE = "https://www.tinkoff.ru/business/"


# название модели для получения эмебддингов

EMB_MODEL_NAME = "cointegrated/LaBSE-en-ru"
# если модель уже скачена и сохранена в папку
if (os.path.exists("models--cointegrated--LaBSE-en-ru/snapshots/cf0714e606d4af551e14ad69a7929cd6b0da7f7e/")):
    print("Модель для эмбеддингов текстов уже существует")
    EMB_MODEL_NAME = "models--cointegrated--LaBSE-en-ru/snapshots/cf0714e606d4af551e14ad69a7929cd6b0da7f7e/"        # это папка
else:
    print("Модель для эмбеддингов текстов НЕ скачена")

# папка с файлами векторной БД
DB_FAISS = "data/dataset.faiss"

# для доступа к YandexGPT
YGPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
YGPT_URI = "ds://bt3842m74v5ui1ml8usdak"

TEMPERATUE = 0.5        # нужна для всех моделей  
# что писать LLM (пока для YandexGPT) перед контекстом
PRE_PROMPT = ("Ты бот-помощник для помощи клиентам банка Тинькофф. Отвечай только по базе знаний Тинькофф, если ответа "
              "нет, то отвечай, что не знаешь, чтобы не ввести в заблуждение. В конце обязательно вставляй ссылку на "
              "статью откуда узнал. Дополнительная информация для ответа. По возможности разделяй информация по пунктам.")


def init_emb_model(model_name:str, model_kwargs:dict, encode_kwargs:dict):
    """Скачивает (если нужно) языковую модель для эмбеддингов, возвращает её"""
    print("Создание\скачивание модели эмбеддингов текстов... ", end="")
    # загрузка модели эмбеддингов
    Embeddings_maker = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
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
    print(f"текстов: {len(texts)}")  # при максимальном размере вопроса в токенах 384, разбивать вопросы на части не пришлось.
    return texts


def get_context(user_request: str, db:FAISS, top:int):
    """Получить контекст для вопроса (top - число документов) используя БД db
    @param user_request: исходный запрос пользователя
    @param db - объект векторной БД
    @param top - сколько похожих объектов извлекать?
    #@return контекст (текст со встроенными ссылками), список ссылок
    """
    # Получить векторный контекст и расстояние между векторами
    context = db.similarity_search_with_score(user_request, k=top)
    print(context)
    # Ближайший из предоставленных
    nearest = min(context, key=lambda x: x[1])
    # Если ближайший вектор очень близок, то это в точности тот вопрос, значит можно вернуть его описание и ссылку на статью
    # Статус 0 - ответ не очевиден, стоит обратиться к модели
    # Статус 1 - ответ очевиден, можно выдать его, чтобы не загружать модель лишний раз
    # Статус 2 - ответ слишком не очевиден и непонятно как дать на него ответ исходя из базы знаний Тинькофф
    if nearest[1] < 0.05: # Взяли среднее значение расстояния из 100 близких вопросов
        return 1, f"{nearest[0].metadata['description']}. Подробнее по ссылке {nearest[0].metadata['url']}", [nearest[0].metadata['url']]
    if nearest[1] > 0.71: # Взяли среднее значение расстояния из 100 плохих вопросов
        return 2, "", []
    # идекс 0 - документ
    # идекс 1 - похожесть
    context_text = "\n\n".join([ "Вопрос: " + text[0].page_content +
                            "\nОтвет: " +  text[0].metadata['description'] +
                           "\nСсылка: " + text[0].metadata['url'] for text in context])
    links = [text[0].metadata['url'] for text in context]
    # description - ответ
    # title - вопрос
    return 0, context_text, links


def init_DB():
    """Инициализирует всё, что необходимо для работы базы знаний (модель эмбеддингов текстов, векторная БД);
    Векторная БД загружается из файлов"""
    # загрузка модели эмбеддингов
    Embeddings_maker = init_emb_model(model_name=EMB_MODEL_NAME,
                                               model_kwargs={'device': 'cpu'},
                                               encode_kwargs={'normalize_embeddings': False})
    # создаем хранилище
    print("создаем хранилище... ", end="")
    DB = FAISS.load_local(folder_path=DB_FAISS, embeddings=Embeddings_maker,
                          allow_dangerous_deserialization=True  # да, я уверен, что в файле нет вредоносного кода
                          )
    print("Готово")
    return DB


def get_Answer_from_YAGPT(text):
    """Получить ответ от YandexGPT. Нужно задать Api-Key и x-folder-id (см. код внизу файла)"""
    global DB   # База знаний
    # Получаем состояние ответа, ближайший контекст для RAG, ссылки
    state, context, links = get_context(text, DB, top=2)
    # Если состояние 0, то ответ не очевидный, но можно его собрать при помощи модели
    if state == 0:
        req = {
                "modelUri": YGPT_URI,
                "completionOptions": {
                    "stream": False,
                    "temperature": TEMPERATUE,
                    "maxTokens": "2000"
                },
                "messages": [
                    {
                    "role": "system",
                    "text": f"{PRE_PROMPT}:{context}"
                    },
                    {
                    "role": "user",
                    "text": text
                    }
                ]
        }
        headers = {"Authorization" : "Api-Key " + API_KEY, "x-folder-id": X_FOLDER_ID, }
        res = requests.post(YGPT_URL, headers=headers, json=req)
        print(res)
        res = res.json()
        if 'result' in res:
            return res['result']['alternatives'][0]['message']['text'], links
        else:
            return  "У нас технические неполадки. Исправляем. Напишите нам позже. А пока можете посмотреть ответ на Ваш вопрос по ссылкам: ", links
    # Если состояние 1, то ответ очевидный и можно его вернуть из базы знаний без вмешательства модели
    elif state == 1:
        return context, links
    # Если состояние 2, то ответ не находится в базе знаний
    elif state == 2:
        return (f"У меня нет ответа на этот вопрос. Если вопрос связан с работой банка, "
                f"то обратитесь в поддержку по номеру телефона {SUPPORT_PHONE} или воспользуйтесь справкой на странице {MAIN_HELP_PAGE}"), []


DB = init_DB()

# загрузка данных для доступа к YandexGPT, (если она используется)
API_KEY, X_FOLDER_ID = [ line.strip() for line in open("ygpt_secret.txt").readlines() ]

