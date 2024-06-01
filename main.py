"""
Главный файл приложения. Криво добавлен сервер (см. todo ниже)
Для простоты отвечает только контекстными документами.
Важно чтобы этот файл назывался main.py!
"""

from fastapi import FastAPI, HTTPException
from models import Request, Response, HTTPValidationError
import uvicorn
import chat_bot as chat_bot
import argparse

print("Создание сервера")
app = FastAPI(title="Assistant API", version="0.1.0")

HOST_DEFAULT = "localhost"
PORT_DEFAULT = 60004


@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})
async def assist(request: Request):
    # global chat_bot.DB
    print(f"Вопрос: {request.query}")
    text, links = chat_bot.get_Answer_from_YAGPT(request.query)
    print(f"Ответ: {text}")
    return Response(text=text, links=links)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск сервера")
    parser.add_argument('--host', type=str, default=HOST_DEFAULT, help='Хост для запуска сервера')
    parser.add_argument('--port', type=int, default=PORT_DEFAULT, help='Порт для запуска сервера')
    args = parser.parse_args()
    # todo: help

    print("Запуск сервера")
    print(f"Запуск на {args.host}:{args.port}")
    uvicorn.run("main:app", host=args.host, port=args.port, reload=True)


# 1. Запуск ollama serve
# 2. Запуск этого файла: python3 main.py
# 3. проверка
# curl -X POST -H 'Content-Type: application/json' -d '{"query":"Как мне получить кредит?"}'  http://0.0.0.0:60004/assist
# curl -X POST -H 'Content-Type: application/json' -d '{"query":"Как пропатчить KDE под FreeBSD?"}'  http://0.0.0.0:60004/assist
