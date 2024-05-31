"""
Главный файл приложения. Криво добавлен сервер (см. todo ниже)
Отвечает только контекстными документами.
Важно чтобы этот файл назывался main.py!
"""

from fastapi import FastAPI, HTTPException
from models import Request, Response, HTTPValidationError
import uvicorn
import chat_bot


DB = chat_bot.init_DB()  # todo: это запускается три раза. Как исправить?

app = FastAPI(title="Assistant API", version="0.1.0")


@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})
async def assist(request: Request):
    global DB
    context = chat_bot.get_context(request.query, DB, top=2)
    return Response(text=f"Processed query: {context}", links=["http://example.com"])


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
