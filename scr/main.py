from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from model import predict_1, predict_2


app = FastAPI()

# pydantic models
class StockIn(BaseModel):
    text: str

class StockOut(BaseModel):  #StockIn
    forecast: list


@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):

    my_text = payload.text

    sentence_one = [my_text]

    questions_candidates = predict_1(sentence_one)

    output_dict_list = predict_2(questions_candidates)


    if not output_dict_list:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {'forecast': output_dict_list}   # 'text': prediction_list, 
    return response_object
