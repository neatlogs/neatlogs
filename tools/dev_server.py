import json
import logging

from fastapi import FastAPI, Request

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/api/data/v3")
async def log_sink(request: Request):
    print(f"Request Received: {request}")
    data = await request.json()
    trace = data["dataDump"]
    logger.info(f"Trace: {json.dumps(trace, indent=2)}")
    return {"message": "Hello, World!"}


@app.get("/")
async def root():
    return {"message": "Hello, World!"}
