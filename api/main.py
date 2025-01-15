from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import psycopg2


app = FastAPI()

class TrainingParams(BaseModel):
    user_id: int
    parameters: dict
    

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")
print('This is databse connection',DATABASE_URL)

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

@app.get("/save_tr_result/")
def save_tr_result():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM invoices WHERE status = 'pending'")
    params = cursor.fetchall()
    conn.close()
    #parse the params
    print('This is params',params[0])
    return {"params": params}




@app.post("/webhook/")
async def receive_webhook(data: TrainingParams):
    try:
        # Assuming 'process_training' is a function that processes the training data
        process_training(data.user_id, data.parameters)
        return {"status": "Success", "message": "Training started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_training(user_id: int, params: dict):
    print(f"Starting training for user {user_id} with parameters {params}")
    # Add the logic to initiate the training process here