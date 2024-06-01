"""
Главный файл приложения. Криво добавлен сервер (см. todo ниже)
Для простоты отвечает только контекстными документами.
Важно чтобы этот файл назывался main.py!
"""

from fastapi import FastAPI, HTTPException
from models import Request, Response, HTTPValidationError
import uvicorn
import ollama
import chat_bot as chat_bot
import requests

print("Создание сервера")
app = FastAPI(title="Assistant API", version="0.1.0")


@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})
async def assist(request: Request):
    # global chat_bot.DB
    print(f"Вопрос: {request.query}")
    text, links = chat_bot.get_Answer_from_YAGPT(request.query)
    print(f"Ответ: {text}")
    return Response(text=text, links=links)


if __name__ == "__main__":
    print("Запуск сервера")
    # uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
    HOST = "localhost"
    PORT = 60004                        
    print(f"Запуск на {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

#
# 1. Запуск ollama serve
# 2. Запуск этого файла
# 3. проверка
# curl -X POST -H 'Content-Type: application/json' -d '{"query":"Как мне получить кредит?"}'  http://0.0.0.0:60004/assist
