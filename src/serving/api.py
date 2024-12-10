from fastapi import FastAPI
from pydantic import BaseModel
from src.modeling.predict import predict

app = FastAPI()

class PredictionRequest(BaseModel):
    player_dmg: float
    player_survive_time: float
    cluster_0: int
    cluster_1: int
    cluster_2: int
    drive_type: int

@app.post("/predict")
def get_prediction(request: PredictionRequest):
    input_data = [[request.player_dmg, request.player_survive_time, request.cluster_0, request.cluster_1, request.cluster_2, request.drive_type]]
    prediction = predict(input_data)
    return {"prediction": prediction}
