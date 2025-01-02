import asyncio
import re
import httpx
from typing import Dict
from unidecode import unidecode
from string import punctuation

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from .classifier import BERTClassifier, get_bert

# import requests
# from starlette.requests import Request
# from starlette.routing import request_response

app = FastAPI()

class ClassificationRequest(BaseModel):
    text: str
    identificador: str
    datetime: str

class ClassificationResponse(BaseModel):
    probabilities: Dict[str, float]
    sentiment: str
    confidence: float

def preProText(text):
    text = text.lower()
    text = re.sub('@[^\s]+', '', text)
    text = unidecode(text)
    text = re.sub('<[^<]+?>', '', text)
    text = ''.join(c for c in text if not c.isdigit())
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = ''.join(c for c in text if c not in punctuation)
    return text

def verTermos(text):
    termos = [
        "suicida", "suicidio", "me matar", "meu bilhete suicida",
        "minha carta suicida", "acabar com a minha vida", "nunca acordar",
        "não consigo continuar", "não vale a pena viver", "pronto para pular",
        "dormir para sempre", "quero morrer", "estar morto",
        "melhor sem mim", "vou me matar", "plano de suicídio", 
        "cansado de viver", "morrer sozinho"
    ]
    return any(term in text for term in termos)

# POST
@app.post("/classifica", response_model=ClassificationResponse)
async def classifica(rqt: ClassificationRequest, model: BERTClassifier = Depends(get_bert)):
    texto = preProText(rqt.text)
    identificador = rqt.identificador
    datetime = rqt.datetime

    url = "http://127.0.0.1:8000/classifica"

    if verTermos(texto):
        try:
            sentiment, confidence, probabilities = model.predict(texto)
            probabilidade = round(float(confidence), 5)
            possibilidade = int(sentiment)
        except Exception as e:
            raise RuntimeError(f"Erro ao realizar a predição: {e}")

    else:
        probabilidade = 0.0
        possibilidade = 0
        sentiment, probabilities = "Neutral", {}

    payload = {
        #'token': token,
        'text': texto,
        'identificador': identificador,
        'probabilidade': probabilidade, 
        'possibilidade': possibilidade,
        'datetime': datetime
    }

    """
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, read=90.0)) as client:
            resposta = await client.post(url, json=payload)
            resposta.raise_for_status()

        print("Conectado com SUCESSO")
    except httpx.HTTPStatusError as e:  # Específico para erros HTTP
        print(f"Erro HTTP: {e.response.status_code}")
        print(f"Detalhes do erro: {e.response.text}")
        raise RuntimeError(f"Erro ao conectar ao servidor remoto: {str(e)}")
    except httpx.RequestError as e:  # Erros relacionados à conexão
        print(f"Erro de requisição: {e}")
        raise RuntimeError(f"Erro ao conectar ao servidor remoto: {str(e)}")
    """

    return ClassificationResponse(
    sentiment=str(sentiment),  # Convertendo para string
    confidence=probabilidade,
    probabilities={str(k): v for k, v in probabilities.items()}
    )

# GET
@app.get("/", response_model=ClassificationResponse)
async def root(text: str = "", model: BERTClassifier = Depends(get_bert)):
    if not text:
        return ClassificationResponse(
            sentiment="Neutral",
            confidence=0.0,
            probabilities={}
        )

    texto = preProText(text)

    if verTermos(texto):
        try:
            sentiment, confidence, probabilities = model.predict(texto)
            probabilidade = round(float(confidence), 5)
            probabilities = int(sentiment)
        except Exception as e:
            raise RuntimeError(f"Erro ao realizar a predição: {e}")
    else:
        probabilidade = 0.0
        probabilities = 0
        sentiment, probabilities = "Neutral", {}

    return ClassificationResponse(
        sentiment=sentiment,
        confidence=probabilidade,
        probabilities=probabilities
    )

@app.get("/favicon.ico")
async def favicon():
    return {"message": "Favicon não configurado."}
