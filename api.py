from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from src.inference import predict_weekly_occupancy
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="Dan Hotels AI - Occupancy DSS", version="1.0")

class WeeklyRequest(BaseModel):
    hotel_id: str
    start_date: str
    past_14_days_occupancy: List[float]

@app.post("/predict/weekly")
def predict_weekly(request: WeeklyRequest):
    if request.hotel_id not in ['DT', 'DJ']:
        raise HTTPException(status_code=400, detail="Invalid property ID. Use 'DT' or 'DJ'.")
    
    if len(request.past_14_days_occupancy) != 14:
        raise HTTPException(status_code=400, detail="Payload requires exactly 14 trailing occupancy rates.")
        
    try:
        safe_date = pd.to_datetime(request.start_date, dayfirst=True).strftime('%Y-%m-%d')
        
        predictions = predict_weekly_occupancy(safe_date, request.hotel_id, request.past_14_days_occupancy)
            
        return {
            "property_code": request.hotel_id,
            "forecast_start_date": safe_date,
            "weekly_forecast": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))