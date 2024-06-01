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


DB = chat_bot.init_DB()  # todo: это запускается три раза. Как исправить?

app = FastAPI(title="Assistant API", version="0.1.0")


@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})
async def assist(request: Request):
    global DB
    print(f"Вопрос: {request.query}")
    context = chat_bot.get_context(request.query, DB, top=2)
    response = ollama.chat(model=chat_bot.LLM_NAME, messages=[
        {
            'role': 'user',
            'content': f'Дай развёрнутый и как можно более точный ответ на вопрос пользователя. '
                       f'Для ответа используй дополнительную информацию (FAQ). Приведи релевантные ссылки. '
                       f'\nВопрос: {request}.\n Дополнительная информация (FAQ): {context}', }],
                           stream=False
                           )
    print(f"Ответ: {response['message']['content']}")
    return Response(text=f"Processed query: {response['message']['content']}", links=["http://example.com"])


if __name__ == "__main__":
    print("Запуск сервера")
    # uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
    HOST = "0.0.0.0"
    PORT = 60003
    print(f"Запуск на {HOST}:{PORT}")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
