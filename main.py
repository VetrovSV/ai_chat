"""
Главный файл приложения

Дописывать обработки запроса по API сюда
"""
import chat_bot
import ollama
# класс для хранения данных как в векторной БД?. Используется для быстрого поиска подходящего контекста по запросу
from langchain_community.vectorstores import FAISS


# название модели для получения эмебддингов
EMB_MODEL_NAME = "cointegrated/LaBSE-en-ru"
# название большой языковойй модели
LLM_NAME = "dimweb/ilyagusev-saiga_llama3_8b:Q6_K"
LLM_NAME = "gemma:2b"


# загрузка модели эмбеддингов
Embeddings_maker = chat_bot.init_emb_model( model_name= EMB_MODEL_NAME,
                                            model_kwargs = {'device': 'cpu'}, encode_kwargs = {'normalize_embeddings': False} )
# попробовать нормализацию эмбеддингов?


Texts = chat_bot.load_dataset( filename_json = "data/dataset.json", embeddings_maker=Embeddings_maker)
# создаем хранилище
print("создаем хранилище... ", end="")
DB = FAISS.from_documents(Texts, Embeddings_maker)
print("Создано")
# db.as_retriever()           # ???
# todo: тут нужно сохранить БД в отдельное место. Иначе создание занимает пару минут


request = input("Вопрос: ")

context = chat_bot.get_context(request, DB, top = 5)
# todo: где-то тут нужно уметь определить, что ответить невозможно. Выдывать в качестве ответа контакты или инфу для
#  связи со специалистом
response = ollama.chat(model=LLM_NAME, messages=[
  {
    'role': 'user',
    'content': f'Дай развёрнутый и как можно более точный ответ. Для ответа используй дополнительную информацию.\nВопрос: {request}.\n Дополнительная информация: {context}',}],
    stream = True
)


print("\n\n Ответ:")
for chunk in response:
  print(chunk['message']['content'], end='', flush=True)