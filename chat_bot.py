import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
# будет разбивать тексты на части, чтобы делать их них эмбеддинги
from langchain.text_splitter import RecursiveCharacterTextSplitter

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