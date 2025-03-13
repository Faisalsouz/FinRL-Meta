# main.py
import os
import sys
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_meta.custom_crypto_paper_trading import AlpacaPaperTradingCryptoLive
from my_meta.single_asset_training_ppo import train
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppose these are your existing DRL functions:
# from your_drl_logic import train, start_paper_trading_crypto_live

app = FastAPI()
# Configure CORS
origins = [
    "http://localhost:3000",  # Add your frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##########################
# 1) Pydantic models for input
##########################

class TrainRequest(BaseModel):
    start_date: str  = Field(..., example="2025-01-01")
    end_date: str    = Field(..., example="2025-02-01")
    ticker_list: list[str] = Field(..., example=["BTCUSDT","ETHUSDT"])
    data_source: str  = Field("binance", example="binance")
    time_interval: str= Field("15Min", example="1Min")
    technical_indicator_list: list[str] = Field(..., example=["macd","rsi"])
    drl_lib: str      = Field("elegantrl", example="elegantrl")
    env: str          = Field("CryptoTradingEnv", example="CryptoTradingEnv")
    model: str   = Field("ppo", example="ppo")
    # add any other training parameters like break_step, cwd, etc.
    cwd: str          = Field("./papertrading_crypto", example="./papertrading_crypto")
    break_step: float = Field(1e5, example=1e5)

class PaperTradingRequest(BaseModel):
    ticker_list: list[str]      = Field(..., example=["BTCUSD","ETHUSD","SOLUSD"])
    time_interval: str          = Field("5Min", example="5Min")
    drl_lib: str                = Field("elegantrl", example="elegantrl")
    agent: str                  = Field("ppo", example="ppo")
    cwd: str                    = Field("./papertrading_crypto", example="./papertrading_crypto")
    net_dim: list[int]          = Field(..., example=[128,64]) 
    state_dim: int              = Field(22, example=22)
    action_dim: int             = Field(3, example=3)
    API_KEY: str                = Field(..., example="PK...")
    API_SECRET: str             = Field(..., example="...")
    API_BASE_URL: str           = Field("https://paper-api.alpaca.markets", example="https://paper-api.alpaca.markets")
    tech_indicator_list: list[str] = Field(..., example=["macd","rsi","cci","dx"])
    max_stock: float            = Field(100.0, example=100.0)

##########################
# 2) Endpoints
##########################

@app.post("/train")
def train_endpoint(req: TrainRequest):
    """
    Endpoint to trigger the DRL training. 
    The body of the request must match `TrainRequest`.
    """
    logs = []
    
    logger.info("Received training request")
    logs.append("Received training request")
    logger.info(f"Request data: {req.dict()}")
    logs.append(f"Request data: {req.dict()}")

    # Convert the pydantic model to a dictionary or pass directly
    train(
        start_date=req.start_date,
        end_date=req.end_date,
        ticker_list=req.ticker_list,
        data_source=req.data_source,
        time_interval=req.time_interval,
        technical_indicator_list=req.technical_indicator_list,
        drl_lib=req.drl_lib,
        env=req.env,  # or string referencing if you prefer 
        model_name=req.model,
        cwd=req.cwd,
        break_step=req.break_step,
    )
    
    logger.info("Training triggered")
    logs.append("Training triggered")
    
    # For demonstration, just returning the request content:
    return {
        "message": "Training triggered!",
        "received": req.dict(),
        "logs": logs
    }


@app.post("/paper_trade")
def paper_trade_endpoint(req: PaperTradingRequest):
    """
    Endpoint to start live paper trading on Alpaca with your DRL agent.
    """
    logger.info("Received paper trading request")
    logger.info(f"Request data: {req.dict()}")
  

    crp_paper_trading = AlpacaPaperTradingCryptoLive(
        ticker_list=req.ticker_list,
        time_interval=req.time_interval,
        drl_lib=req.drl_lib,
        agent=req.agent,
        cwd=req.cwd,
        net_dim=req.net_dim,
        state_dim=req.state_dim,
        action_dim=req.action_dim,
        API_KEY=req.API_KEY,
        API_SECRET=req.API_SECRET,
        API_BASE_URL=req.API_BASE_URL,
        tech_indicator_list=req.tech_indicator_list,
        max_stock=req.max_stock
    )
    
    logger.info("Starting paper trading")
    crp_paper_trading.run()
    logger.info("Paper trading started successfully")

    return {
        "message": "Paper trading started!",
        "received": req.dict(),
        "logs": [
            "Received paper trading request",
            f"Request data: {req.dict()}",
            "Starting paper trading",
            "Paper trading started successfully"
        ]
    }

##########################
# 3) Launch with uvicorn
##########################

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




























# old next.js code #####################
# #####################

# # from fastapi import FastAPI, HTTPException, Request
# from pydantic import BaseModel
# import os
# import psycopg2


# app = FastAPI()

# class TrainingParams(BaseModel):
#     user_id: int
#     parameters: dict
    

# app = FastAPI()

# DATABASE_URL = os.getenv("DATABASE_URL")
# print('This is databse connection',DATABASE_URL)

# def get_db_connection():
#     conn = psycopg2.connect(DATABASE_URL)
#     return conn

# @app.get("/save_tr_result/")
# def save_tr_result():
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM invoices WHERE status = 'pending'")
#     params = cursor.fetchall()
#     conn.close()
#     #parse the params
#     print('This is params',params[0])
#     return {"params": params}




# @app.post("/webhook/")
# async def receive_webhook(data: TrainingParams):
#     try:
#         # Assuming 'process_training' is a function that processes the training data
#         process_training(data.user_id, data.parameters)
#         return {"status": "Success", "message": "Training started"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# def process_training(user_id: int, params: dict):
#     print(f"Starting training for user {user_id} with parameters {params}")
#     # Add the logic to initiate the training process here