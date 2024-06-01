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
    #@return
    """
    # todo: добавлять ссылку
    # todo: сделать отбор документов для контекста на основе порогового расстояния?
    context = db.similarity_search_with_score(user_request, k=top)
    # print(context)
    # идекс 0 - документ
    # идекс 1 - похожесть
    context = "\n\n".join([ "Вопрос: " + text[0].page_content +
                            "\nОтвет: " +  text[0].metadata['description'] +
                           "\nСсылка: " + text[0].metadata['url'] for text in context])
    # description - ответ
    # title - вопрос
    # todo: контролировать размер контекста, чтобы он влезал в промпт LLM
    # print(context)
    return context


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
    return DB