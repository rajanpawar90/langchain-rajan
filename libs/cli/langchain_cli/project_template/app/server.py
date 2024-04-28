from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.serving.fastapi import add_routes

app = FastAPI()

