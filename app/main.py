import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from app.db import engine, create_db_and_tables, PredictionsTickets
from app.utils import preprocessing_fn
from sqlmodel import Session, select
from enum import Enum

app = FastAPI(title = "FastAPI, Docker, and Grafana")
global label_mapping

label_mapping = {
    "0": "Bank Account Services",
    "1": "Credit Report or Prepaid Card",
    "2": "Mortgage/Loan",
}

class Sentence(BaseModel):
    client_name : str
    text: str

class ProcessTextRequestModel(BaseModel):
    sentences: list[Sentence]

@app.post("/predict") 
async def read_root(data: ProcessTextRequestModel):

    session = Session(engine)

    model = joblib.load("model.pkl")

    preds_list = []

    