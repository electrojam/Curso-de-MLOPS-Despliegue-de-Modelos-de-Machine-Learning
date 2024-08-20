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

    for sentence in data.sentences:  #iteramos cada dado de estructura request, dat hereda  lista
        processed_data_vectorized = preprocessing_fn(sentence.text)
        X_dense = [sparse_matrix.toarray() for sparse_matrix in processed_data_vectorized]  #generamos matriz dispersa
        X_dense = np.vstack(X_dense)  #ponemos matriz dispersa de forma vertical

        preds = model.predict(X_dense)  #generamos predicciones
        decoded_predictions = label_mapping[str(preds[0])]  #extrames la predicción en la primer posición

        prediction_ticket =  PredictionsTickets(  #creamos objeto con esas predicciones
            client_name = sentence.client_name,  #almacenamos campo client_name
            prediction = decoded_predictions  #almacenamos la predicción
        )

        print(prediction_ticket)  #imprimimos objeto

        preds_list.append({  #almacenamos todas las predicciones en la lista vacía creada 
            "client_name": sentence.client_name,
            "prediction": decoded_predictions
        })

        session.add(prediction_ticket)  #almacenamos todas las predicciones en la base de datos

    session.commit()
    session.close()


    return {"predictions": preds_list}

@app.on_event("startup")  #inicializar db cuando la app esté levantada
async def startup():
    create_db_and_tables()
