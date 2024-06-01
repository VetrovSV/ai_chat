from fastapi import FastAPI, HTTPException
from models import Request, Response, HTTPValidationError
import uvicorn


app = FastAPI(title="Assistant API", version="0.1.0")


@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})
async def assist(request: Request):
    return Response(text=f"Processed query: {request.query}", links=["http://example.com"])


if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
