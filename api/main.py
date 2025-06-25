# main.py
import os
import sys
from fastapi import FastAPI, Body
import threading
import pandas as pd
import os
from pydantic import BaseModel, Field
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_meta.custom_crypto_paper_trading import AlpacaPaperTradingCryptoLive
from my_meta.single_asset_training_ppo import train,test
import logging
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppose these are your existing DRL functions:
# from your_drl_logic import train, start_paper_trading_crypto_live

app = FastAPI()
# Configure CORS
origins = [
    "http://localhost:3000",
    "https://b17a-77-22-186-73.ngrok-free.app" # Add your frontend URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

crp_paper_trading = None

@app.post("/stop_paper_trade")
def stop_paper_trade_endpoint():
    """
    Endpoint to stop the live paper trading loop.
    """
    global crp_paper_trading
    logger.info("Received stop paper trading request")
    
    if crp_paper_trading:
        crp_paper_trading.stop()
        logger.info("Paper trading stopped successfully")
        return {
            "message": "Paper trading stopped!",
            "logs": [
                "Received stop paper trading request",
                "Paper trading stopped successfully"
            ]
        }
    else:
        logger.error("Paper trading instance not found")
        return {
            "message": "Paper trading instance not found!",
            "logs": [
                "Received stop paper trading request",
                "Paper trading instance not found"
            ]
        }

@app.get("/fetch_logs")
def fetch_logs():
    """
    Endpoint to fetch the log file contents.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trade_log_path = script_dir + "/papertrading_crypto/paper_trading_trade_logs.csv"
    performance_csv_path = script_dir + "/papertrading_crypto/performance_data.csv"
    
    try:
        trade_df = pd.read_csv(trade_log_path)
        performance_df = pd.read_csv(performance_csv_path)
        
        
        # Extract performance metrics from the performance CSV
        performance_data = performance_df.to_dict(orient='records')[0]

        return {
            "date": performance_data.get("date", ""),
            "atr_avg": performance_data.get("atr_avg", 0.0),
            "profit_loss": performance_data.get("profit_loss", 0.0),
            "equity_change": performance_data.get("equity_change", 0.0),
            "avg_trade_per_day": performance_data.get("avg_trade_per_day", 0.0),
            "win_trades": performance_data.get("win_trades", 0),
            "lost_trades": performance_data.get("lost_trades", 0),
            "timeout_trades": performance_data.get("timeout_trades", 0),
            "avg_win": performance_data.get("avg_win", 0.0),
            "avg_loss": performance_data.get("avg_loss", 0.0),
            "stop_loss_stats": performance_data.get("stop_loss_stats", 0.0)
        }
    except Exception as e:
        return {"error": str(e)}
    
    log_file_path = script_dir + "/papertrading_crypto/paper_trading.log"
    return FileResponse(log_file_path, media_type='text/plain')

@app.get("/fetch_logs_train")
def fetch_logs_train():
    """
    Endpoint to fetch the training log file contents.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_file_path = script_dir + "/papertrading_crypto/training.log"
    return FileResponse(log_file_path, media_type='text/plain')
@app.get("/list_artifacts")
def list_artifacts():
    """
    Endpoint to list the contents of the papertrading_crypto directory.
    """
    directory_path = "/home/souz_wsl/finrl_proj/FinRL_Meta/api/papertrading_crypto"
    try:
        files = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                files.append({
                    "name": filename,
                    "created_at": os.path.getctime(file_path),
                    "modified_at": os.path.getmtime(file_path),
                    "path": file_path
                })
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}

@app.get("/download_artifact")
def download_artifact(path: str):
    """
    Endpoint to download a specific artifact file.
    """
    return FileResponse(path, media_type='application/octet-stream', filename=os.path.basename(path))

@app.get("/fetch_performance_analysis")
def fetch_performance_analysis(start_date: str = None, end_date: str = None):
    """
    Endpoint to fetch performance analysis data.
    """
    import pandas as pd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    performance_csv_path = script_dir + "/papertrading_crypto/paper_trading_performance_data.csv"
    
    try:
        performance_df = pd.read_csv(performance_csv_path)
        
        if start_date:
            performance_df = performance_df[performance_df['date'] >= start_date]
        if end_date:
            performance_df = performance_df[performance_df['date'] <= end_date]
        
        # Extract performance metrics from the filtered performance CSV
        performance_data = performance_df.to_dict(orient='records')
        print(performance_data)  # Debugging line to check the data structure

        # Initialize lists for each metric
        dates = []
        atr_avg_list = []
        profit_loss_list = []
        equity_change_list = []
        avg_trade_per_day_list = []
        win_trades_list = []
        lost_trades_list = []
        timeout_trades_list = []
        avg_win_list = []
        avg_loss_list = []
        stop_loss_stats_list = []

        # Populate lists with values from the filtered data
        for data in performance_data:
            dates.append(data.get("date", ""))
            atr_avg_list.append(data.get("atr_avg", 0.0))
            profit_loss_list.append(data.get("profit_loss", 0.0))
            equity_change_list.append(data.get("equity_change", 0.0))
            avg_trade_per_day_list.append(data.get("avg_trade_per_day", 0.0))
            win_trades_list.append(data.get("win_trades", 0))
            lost_trades_list.append(data.get("lost_trades", 0))
            timeout_trades_list.append(data.get("timeout_trades", 0))
            avg_win_list.append(data.get("avg_win", 0.0))
            avg_loss_list.append(data.get("avg_loss", 0.0))
            stop_loss_stats_list.append(data.get("stop_loss_stats", 0.0))

        # Format the response for the frontend
        response = {
            "dates": dates,
            "atr_avg": atr_avg_list,
            "profit_loss": profit_loss_list,
            "equity_change": equity_change_list,
            "avg_trade_per_day": avg_trade_per_day_list,
            "win_trades": win_trades_list,
            "lost_trades": lost_trades_list,
            "timeout_trades": timeout_trades_list,
            "avg_win": avg_win_list,
            "avg_loss": avg_loss_list,
            "stop_loss_stats": stop_loss_stats_list
        }

        logger.info(f"Performance data response: {response}")  # Add logging
        return response
    except Exception as e:
        logger.error(f"Error fetching performance data: {str(e)}")  # Add logging
        return {"error": str(e)}

@app.get("/fetch_logs_test")
def fetch_logs_test():
    """
    Endpoint to fetch the training log file contents.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    log_file_path = script_dir + "/papertrading_crypto/test.log"
    return FileResponse(log_file_path, media_type='text/plain')

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
    env: str          = Field("CryptoTradingEnv", example="CryptoTradingEnv")  # Keep as string
    model: str   = Field("ppo", example="ppo")   
    break_step: float = Field(1e5, example=1e5)
    initial_capital: float = Field(10000, example=10000)
    tp_multiplier: float = Field(2, example=2)
    sl_multiplier: float = Field(5, example=5)
    atr_window: int = Field(14, example=14)
    max_trade_duration: int = Field(20, example=20)
    gpu_id: int = Field(0, example=0)
    erl_params: dict = Field(..., example={
        "learning_rate": 3e-6,
        "batch_size": 512,
        "gamma": 0.99,
        "seed": 312,
        "net_dimension": [256, 128, 64, 32],
        "target_step": 5000,
        "eval_gap": 30,
        "eval_times": 1,
        "ratio_clip": 0.5,
        "lambda_gae_adv": 0.95,
        "lambda_entropy": 0.01
    })

class PaperTradingRequest(BaseModel):
    ticker_list: list[str]      = Field(..., example=["BTCUSD","ETHUSD","SOLUSD"])
    time_interval: str          = Field("5min", example="5min")
    drl_lib: str                = Field("elegantrl", example="elegantrl")
    agent: str                  = Field("ppo", example="ppo")
    net_dim: list[int]          = Field(..., example=[128,64]) 
    state_dim: int              = Field(22, example=22)
    action_dim: int             = Field(3, example=3)
    API_KEY: str                = Field(..., example="PK...")
    API_SECRET: str             = Field(..., example="...")
    cwd: str                   = Field(..., example="papertrading_crypto")
    API_BASE_URL: str           = Field("https://paper-api.alpaca.markets", example="https://paper-api.alpaca.markets")
    tech_indicator_list: list[str] = Field(..., example=["macd","rsi","cci","dx"])
    max_stock: float            = Field(100.0, example=100.0)
    tp_multiplier: float        = Field(2, example=2)
    sl_multiplier: float        = Field(5, example=5)
    atr_window: int             = Field(14, example=14)
    max_trade_duration: int     = Field(20, example=20)
    actor_filename: str = Field("best_actor.pth", example="best_actor.pth or latest_actor.pth or ...")

##########################
# 2) Endpoints
##########################

from my_meta.custom_crypto_env import CryptoTradingEnv  # Ensure this import is present

ENV_CLASSES = {
    "CryptoTradingEnv": CryptoTradingEnv,
    # Add other environment classes here if needed
}

@app.post("/train")
def train_endpoint(req: TrainRequest):
    logs = []
    
    logger.info("Received training request")
    logs.append("Received training request")
    logger.info(f"Request data: {req.dict()}")
    logs.append(f"Request data: {req.dict()}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd_path = script_dir + "/papertrading_crypto"
    if not os.path.exists(cwd_path):
        os.makedirs(cwd_path)
        logger.info(f"Created directory: {cwd_path}")
        logs.append(f"Created directory: {cwd_path}")
    else:
        logger.info(f"Directory already exists: {cwd_path}")
        logs.append(f"Directory already exists: {cwd_path}")
    
    # Convert the env string to the actual class object
    env_class = ENV_CLASSES.get(req.env)
    if env_class is None:
        logger.error(f"Environment class '{req.env}' not found")
        return {
            "message": f"Environment class '{req.env}' not found!",
            "logs": logs
        }

    # Convert the pydantic model to a dictionary or pass directly
    train(
        start_date=req.start_date,
        end_date=req.end_date,
        ticker_list=req.ticker_list,
        data_source=req.data_source,
        time_interval=req.time_interval,
        technical_indicator_list=req.technical_indicator_list,
        drl_lib=req.drl_lib,
        env=env_class,  # Pass the class object instead of a string
        model_name=req.model,
        erl_params=req.erl_params,  # Pass ERL parameters
        cwd=cwd_path,
        break_step=req.break_step,
        gpu_id=req.gpu_id,
        initial_capital=req.initial_capital,
        tp_multiplier=req.tp_multiplier,
        sl_multiplier=req.sl_multiplier,
        atr_window=req.atr_window,
        max_trade_duration=req.max_trade_duration,
        log_file_path=f"{cwd_path}/training.log",  
    )
    
    logger.info("Training triggered")
    logs.append("Training triggered")
    
    # For demonstration, just returning the request content:
    return {
        "message": "Training triggered!",
        "received": req.dict(),
        "logs": logs
    }


def start_paper_trading(req: PaperTradingRequest):
    global crp_paper_trading
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd_path = os.path.join(script_dir, req.cwd)

    crp_paper_trading = AlpacaPaperTradingCryptoLive(
        ticker_list=req.ticker_list,
        time_interval=req.time_interval,
        drl_lib=req.drl_lib,
        agent=req.agent,
        cwd=cwd_path,
        net_dim=req.net_dim,
        state_dim=req.state_dim,
        action_dim=req.action_dim,
        API_KEY=req.API_KEY,
        API_SECRET=req.API_SECRET,
        API_BASE_URL=req.API_BASE_URL,
        tech_indicator_list=req.tech_indicator_list,
        max_stock=req.max_stock,
        tp_multiplier=req.tp_multiplier,
        sl_multiplier=req.sl_multiplier,
        atr_window=req.atr_window,
        max_trade_duration=req.max_trade_duration,
        log_file_path=cwd_path + "/paper_trading.log",
        actor_filename=req.actor_filename  
    )
    
    logger.info("Starting paper trading")
    crp_paper_trading.run()
    logger.info("Paper trading started successfully")

@app.post("/paper_trade")
def paper_trade_endpoint(req: PaperTradingRequest):
    """
    Endpoint to start live paper trading on Alpaca with your DRL agent.
    """
    logs = []
    try:
        logger.info("Received paper trading request")
        logs.append("Received paper trading request")
        logger.info(f"Request data: {req.dict()}")
        logs.append(f"Request data: {req.dict()}")

        # Start paper trading in a separate thread
        paper_trading_thread = threading.Thread(target=start_paper_trading, args=(req,))
        paper_trading_thread.start()

        return {
            "message": "Paper trading started!",
            "received": req.dict(),
            "logs": logs,
            "error": None  # No error occurred
        }
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logs.append(f"Error occurred: {str(e)}")
        return {
            "message": "Failed to start paper trading!",
            "received": req.dict(),
            "logs": logs,
            "error": str(e)  # Include the error message
        }
        
                                                                                                                                                                   
class TestRequest(BaseModel):
    start_date: str  = Field(..., example="2024-10-25")
    end_date: str    = Field(..., example="2025-02-02")
    ticker_list: list[str] = Field(..., example=["BTCUSDT"])
    data_source: str  = Field("binance", example="binance")
    time_interval: str= Field("30m", example="30m")
    technical_indicator_list: list[str] = Field(..., example=["macd","rsi","cci","dx"])
    drl_lib: str      = Field("elegantrl", example="elegantrl")
    env: str          = Field("CryptoTradingEnv", example="CryptoTradingEnv")
    model: str   = Field("ppo", example="ppo")
    net_dimension: list[int] = Field(..., example=[256, 128, 64, 32])
    cwd: str                   = Field(..., example="papertrading_crypto")
    initial_capital: float = Field(10000, example=10000)
    atr_window: int = Field(14, example=14)  # Add ATR window
    tp_multiplier: float = Field(2, example=2)  # Add TP multiplier
    sl_multiplier: float = Field(1, example=1)  # Add SL multiplier
    max_trade_duration: int = Field(20, example=20)
    actor_filename: str = Field("best_actor.pth", example="best_actor.pth or latest_actor.pth or ...")
    
    
# Test endpoint to trigger the test function
@app.post("/test")                                                                                                                                                 
def test_endpoint(req: TestRequest):                                                                                                                               
    logs = []                                                                                                                                                      
                                                                                                                                                                   
    logger.info("Received test request")                                                                                                                           
    logs.append("Received test request")                                                                                                                           
    logger.info(f"Request data: {req.dict()}")                                                                                                                     
    logs.append(f"Request data: {req.dict()}")                                                                                                                     
                                                                                                                                                                   
    script_dir = os.path.dirname(os.path.abspath(__file__))                                                                                                        
    cwd_path = script_dir + "/papertrading_crypto"                                                                                                                 
    if not os.path.exists(cwd_path):                                                                                                                               
        os.makedirs(cwd_path)                                                                                                                                      
        logger.info(f"Created directory: {cwd_path}")                                                                                                              
        logs.append(f"Created directory: {cwd_path}")                                                                                                              
    else:                                                                                                                                                          
        logger.info(f"Directory already exists: {cwd_path}")                                                                                                       
        logs.append(f"Directory already exists: {cwd_path}")                                                                                                       
                                                                                                                                                                   
    env_class = ENV_CLASSES.get(req.env)                                                                                                                           
    if env_class is None:                                                                                                                                          
        logger.error(f"Environment class '{req.env}' not found")                                                                                                   
        return {                                                                                                                                                   
            "message": f"Environment class '{req.env}' not found!",                                                                                                
            "logs": logs                                                                                                                                           
        }                                                                                                                                                          
                                                                                                                                                                   
    test(
        start_date=req.start_date,
        end_date=req.end_date,
        ticker_list=req.ticker_list,
        data_source=req.data_source,
        time_interval=req.time_interval,
        technical_indicator_list=req.technical_indicator_list,
        drl_lib=req.drl_lib,
        env=env_class,
        model_name=req.model,
        net_dimension=req.net_dimension,
        cwd=cwd_path,
        initial_capital=req.initial_capital,
        atr_window=req.atr_window,  # Pass ATR window
        tp_multiplier=req.tp_multiplier,  # Pass TP multiplier
        sl_multiplier=req.sl_multiplier,  # Pass SL multiplier
        log_file_path=f"{cwd_path}/test.log",
        actor_filename=req.actor_filename,
        max_trade_duration=req.max_trade_duration,
          
    )                                                                                                                                                              
                                                                                                                                                                   
    logger.info("Test triggered")                                                                                                                                  
    logs.append("Test triggered")                                                                                                                                  
                                                                                                                                                                   
    return {                                                                                                                                                       
        "message": "Test triggered!",                                                                                                                              
        "received": req.dict(),                                                                                                                                    
        "logs": logs                                                                                                                                               
    }                

##########################
# 3) Launch with uvicorn
##########################

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

























