"""
Главный файл приложения.
"""

from fastapi import FastAPI, HTTPException
from models import Request, Response, HTTPValidationError
import uvicorn
import ollama
import chat_bot as chat_bot
import requests

print("Создание сервера")
app = FastAPI(title="Assistant API", version="0.1.0")

# Запрос assist
@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})
async def assist(request: Request):
    print(f"Вопрос: {request.query}")
    # Получение ответа от чат бота
    text, links = chat_bot.get_Answer_from_YAGPT(request.query)
    print(f"Ответ: {text}")
    return Response(text=text, links=links)


if __name__ == "__main__":
    print("Запуск сервера")
    # uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
    HOST = "0.0.0.0"
    PORT = 60004                        
    print(f"Запуск на {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)