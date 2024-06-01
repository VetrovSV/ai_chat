import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
# будет разбивать тексты на части, чтобы делать их них эмбеддинги
from langchain.text_splitter import RecursiveCharacterTextSplitter

# класс-обёртка для создания эмбеддингов текстов
from langchain_community.embeddings import HuggingFaceEmbeddings

# Класс для хранения данных как в векторной БД?. Используется для быстрого поиска подходящего контекста по запросу
from langchain_community.vectorstores import FAISS

# Класс для отправки GET и POST запроса. Нужен для отправки запроса к YandexGPT Pro
import requests


# название модели для получения эмебддингов
EMB_MODEL_NAME = "cointegrated/LaBSE-en-ru"
# название большой языковойй модели
LLM_NAME = "dimweb/ilyagusev-saiga_llama3_8b:Q6_K"
LLM_NAME = "gemma:2b"
# файл векторной БД с индексами (и чем-то ещё?)
DB_FAISS = "data/dataset.faiss"


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


def get_context(user_request: str, db, top):
    """Получить контекст для вопроса (top - число документов) используя БД db
    @param user_request: исходный запрос пользователя
    @param db - объект векторной БД
    @param top - сколько похожих объектов извлекать?
    #@return контекст (текст со встроенными ссылками), список ссылок
    """
    # todo: добавлять ссылку
    # todo: сделать отбор документов для контекста на основе порогового расстояния?
    context = db.similarity_search_with_score(user_request, k=top)
    print(context)
    nearest = min(context, key=lambda x: x[1])
    # Если ближайший вектор очень близок, то это в точности тот вопрос, значит можно вернуть его описание и ссылку на статью
    # Статус 0 - ответ не очевиден, стоит обратиться к модели
    # Статус 1 - ответ очевиден, можно выдать его, чтобы не загружать модель лишний раз
    # Статус 2 - ответ слишком не очевиден и непонятно как дать на него ответ исходя из базы знаний Тинькофф
    if nearest[1] < 0.05:
        return 1, f"{nearest[0].metadata['description']}. Подробнее по ссылке {nearest[0].metadata['url']}", [nearest[0].metadata['url']]
    if nearest[1] > 10:
        return 2, nearest[0].metadata['description'], [nearest[0].metadata['url']]
    # print(context)
    # идекс 0 - документ
    # идекс 1 - похожесть
    context_text = "\n\n".join([ "Вопрос: " + text[0].page_content +
                            "\nОтвет: " +  text[0].metadata['description'] +
                           "\nСсылка: " + text[0].metadata['url'] for text in context])
    links = [text[0].metadata['url'] for text in context]
    # description - ответ
    # title - вопрос
    # todo: контролировать размер контекста, чтобы он влезал в промпт LLM
    # print(context)
    return 0, context_text, links


def init_DB():
    """для проверки работы сервера"""
    # загрузка модели эмбеддингов
    Embeddings_maker = init_emb_model(model_name=EMB_MODEL_NAME,
                                               model_kwargs={'device': 'cpu'},
                                               encode_kwargs={'normalize_embeddings': False})
    # попробовать нормализацию эмбеддингов?

    # Texts = load_dataset(filename_json="data/dataset.json", embeddings_maker=Embeddings_maker)
    # создаем хранилище
    print("создаем хранилище... ", end="")
    # DB = FAISS.from_documents(Texts, Embeddings_maker)    # вернёт экземпляр FAISS
    # DB.save_local(DB_FAISS)
    # https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.save_local
    DB = FAISS.load_local(folder_path=DB_FAISS, embeddings=Embeddings_maker,
                          allow_dangerous_deserialization=True  # да, я уверен, что в файле нет вредоносного кода
                          )
    print("Готово")
    return DB

def get_Answer_from_YAGPT(text):
    global DB
    state, context, links = get_context(text, DB, top=2)
    if state == 0:
        req = {
                "modelUri": "ds://bt168m74v9ui1ml0upnh",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.1,
                    "maxTokens": "2000"
                },
                "messages": [
                    {
                    "role": "system",
                    "text": f"Ты бот-помощник для помощи клиентам банка Тинькофф. Отвечай только по базе знаний Тинькофф, если ответа нет, то отвечай, что не знаешь, чтобы не ввести в заблуждение. В конце обязательно вставляй ссылку на статью откуда узнал. Дополнительная информация для ответа:{context}"
                    },
                    {
                    "role": "user",
                    "text": text
                    }
                ]
        }
        headers = {"Authorization" : "Api-Key " + 'AQVNz9fpuAtA5fGeDB_rcMEY_byJDgEEPeivMMYn', "x-folder-id": "b1g72uajlds114mlufqi", }
        res = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion", headers=headers, json=req)
        print(res)
        res = res.json()
        return res['result']['alternatives'][0]['message']['text'], links
    elif state == 1:
        return context, links
# нужно перенести в более подходящее место
DB = init_DB()