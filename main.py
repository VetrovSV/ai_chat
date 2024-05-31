"""
Главный файл приложения
"""

import chat_bot

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
Embeddings_maker = HuggingFaceEmbeddings(
    model_name=EMB_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


Texts = chat_bot.load_dataset( filename_json = "data/dataset.json", embeddings_maker=Embeddings_maker)
# создаем хранилище
print("создаем хранилище... ", end="")
db = FAISS.from_documents(Texts, Embeddings_maker)
print("Создано")
# db.as_retriever()           # ???

# todo: тут нужно сохранить БД в отдельное место. Иначе создание занимает пару минут
# # пример использования:

# print(
# db.similarity_search_with_score('Как перевыпустить карту', k = 5 )
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