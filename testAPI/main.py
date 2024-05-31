from fastapi import FastAPI, HTTPException
from models import Request, Response, HTTPValidationError

app = FastAPI(title="Assistant API", version="0.1.0")

@app.post("/assist", response_model=Response, responses={422: {"model": HTTPValidationError}})

async def assist(request: Request):
    return Response(text=f"Processed query: {request.query}", links=["http://example.com"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
