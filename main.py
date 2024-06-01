"""
Главный файл приложения. Криво добавлен сервер (см. todo ниже)
Для простоты отвечает только контекстными документами.
Важно чтобы этот файл назывался main.py!
"""

from fastapi import FastAPI, HTTPException
from models import Request, Response, HTTPValidationError
import uvicorn
import ollama
import chat_bot



print("Создание сервера")
app = FastAPI(title="Assistant API", version="0.1.0")


@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})
async def assist(request: Request):
    # global chat_bot.DB
    print(f"Вопрос: {request.query}")
    context, links = chat_bot.get_context(request.query, chat_bot.DB, top=2)
    # todo: сделать обработку на случай, если сервер ollama недоступен
    response = ollama.chat(model=chat_bot.LLM_NAME, messages=[
        {
            'role': 'user',
            'content': f'Дай развёрнутый и как можно более точный ответ на вопрос пользователя. '
                       f'Для ответа используй дополнительную информацию (FAQ). Приведи релевантные ссылки. '
                       f'\nВопрос: {request}.\n Дополнительная информация (FAQ): {context}', }],
                           stream=False
                           )
    print(f"Ответ: {response['message']['content']}")
    return Response(text=f"Processed query: {response['message']['content']}", links=links)


if __name__ == "__main__":
    print("Запуск сервера")
    # uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
    HOST = "0.0.0.0"
    PORT = 60004                        
    print(f"Запуск на {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

#
# 1. Запуск ollama serve
# 2. Запуск этого файла
# 3. проверка
# curl -X POST -H 'Content-Type: application/json' -d '{"query":"Как мне получить кредит?"}'  http://0.0.0.0:60004/assist
