from fastapi import FastAPI, Query
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from typing import List, Dict
 
# Configuration
DATABASE_URL = "postgresql://postgres:diarms@localhost:5432/postgres"
 
# Connexion PostgreSQL
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
 
# App FastAPI
app = FastAPI(title="API Datamarts Accidents", version="1.0")
 
# Helper
def fetch_paginated_data(table_name: str, offset: int = 0, limit: int = 10) -> List[Dict]:
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT * FROM {table_name} OFFSET :offset LIMIT :limit"),
            {"offset": offset, "limit": limit}
        )
        return [dict(row._mapping) for row in result]
 
 
# Routes GET avec pagination
 
@app.get("/severity-by-city-weather", tags=["Datamart"], summary="Average severity by city, date and weather condition")
def get_top_cities(offset: int = Query(0, ge=0), limit: int = Query(10, le=100)):
    return fetch_paginated_data("dmAcc_severity_by_city_day_weather", offset, limit)
 
 
@app.get("/accidents-hourly-state", tags=["Datamart"], summary="Accident count by hour and state")
def get_weather_stats(offset: int = Query(0, ge=0), limit: int = Query(10, le=100)):
    return fetch_paginated_data("dmAcc_accidents_by_hour_and_state", offset, limit)
 
 
@app.get("/weather-impact-severity", tags=["Datamart"], summary="Impact of weather conditions on accident severity")
def get_severity_distribution(offset: int = Query(0, ge=0), limit: int = Query(10, le=100)):
    return fetch_paginated_data("dmAcc_weather_impact_on_severity", offset, limit)
 
 
@app.get("/temporal-analysis", tags=["Datamart"], summary="Temporal distribution of accidents (year, month, weekday)")
def get_state_ranking(offset: int = Query(0, ge=0), limit: int = Query(10, le=100)):
    return fetch_paginated_data("dmAcc_temporal_analysis_accidents", offset, limit)