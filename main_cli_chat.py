"""
Главный файл приложения. Консольный чат на один запрос.
Файл для примера. Сервер описывается в main.py
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
# файл векторной БД с индексами (и чем-то ещё?)
DB_FAISS = "data/dataset.faiss"


# загрузка модели эмбеддингов
Embeddings_maker = chat_bot.init_emb_model( model_name= EMB_MODEL_NAME,
                                            model_kwargs = {'device': 'cpu'}, encode_kwargs = {'normalize_embeddings': False} )
# попробовать нормализацию эмбеддингов?


Texts = chat_bot.load_dataset( filename_json = "data/dataset.json", embeddings_maker=Embeddings_maker)
# создаем хранилище
print("создаем хранилище... ", end="")
# DB = FAISS.from_documents(Texts, Embeddings_maker)    # вернёт экземпляр FAISS
# DB.save_local(DB_FAISS)
# https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html#langchain_community.vectorstores.faiss.FAISS.save_local
DB = FAISS.load_local(folder_path=DB_FAISS, embeddings=Embeddings_maker,
                                            allow_dangerous_deserialization = True    # да, я уверен, что в файле нет вредоносного кода
                      )
print("Создано")
# db.as_retriever()           # ???
# todo: тут нужно сохранить БД в отдельное место. Иначе создание занимает пару минут


request = input("Вопрос: ")

context = chat_bot.get_context(request, DB, top = 2)

print("Контекст: \n"+context)
# todo: где-то тут нужно уметь определить, что ответить невозможно. Выдавать в качестве ответа контакты или инфу для
#  связи со специалистом
response = ollama.chat(model=LLM_NAME, messages=[
  {
    'role': 'user',
    'content': f'Дай развёрнутый и как можно более точный ответ на вопрос пользователя. '
               f'Для ответа используй дополнительную информацию (FAQ). Приведи релевантные ссылки. '
               f'\nВопрос: {request}.\n Дополнительная информация (FAQ): {context}',}],
    stream = True
)


print("\n\n Ответ:")
for chunk in response:
  print(chunk['message']['content'], end='', flush=True)