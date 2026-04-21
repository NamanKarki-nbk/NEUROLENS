from fastapi import FastAPI
from contextlib import asynccontextmanager

from routes.inference import router
from model_loader import init_models


@asynccontextmanager
async def lifespan(app:FastAPI):
    init_models()
    print("App Started")
    yield
    
    

app = FastAPI(lifespan=lifespan)
app.include_router(router)