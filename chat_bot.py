import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
# будет разбивать тексты на части, чтобы делать их них эмбеддинги
from langchain.text_splitter import RecursiveCharacterTextSplitter

# класс-обёртка для создания эмбеддингов текстов
from langchain_community.embeddings import HuggingFaceEmbeddings


def init_emb_model(model_name:str, model_kwargs:dict, encode_kwargs:dict):
    """Скачивает (если нужно) языковую модель для эмбеддингов, возвращает её"""

    # загрузка модели эмбеддингов
    # model_kwargs = {'device': 'cpu'}
    # encode_kwargs = {'normalize_embeddings': False}
    # сделать более точную настройку параметров, если необходимо
    Embeddings_maker = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return Embeddings_maker


def load_dataset(filename_json: str, embeddings_maker):
    """Загружает датасет, создаёт векторную БД
    @param filename_json -- JSON файл с датасетом вопросов и ответов
    @param embeddings_maker -- языковая модель для создания эмбеддингов
    @return тексты в формате пакета langchain_community
     """
    # загрузка датасета
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
    @param user_request: исхдный запрос пользователя
    @param db - объект векторной БД
    @param top - сколько похожих объектов извлекать?
    #@return
    """
    # todo: добавлять ссылку
    # todo: сделать отбор документов для контекста на основе порогового расстояния?
    context = db.similarity_search_with_score(user_request, k=top)
    context = "\n\n".join([text[0].metadata['description']  for text in context])
    # todo: контролировать размер контекста, чтобы он влезал в промпт LLM
    # print(context)
    return context