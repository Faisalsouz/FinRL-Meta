                                                                                                                                                                                    
"use client";                                                                                                                                                                       
import { useState, useEffect } from "react";                                                                                                                                        
import { Tooltip } from "react-tooltip";                                                                                                                                            
import 'react-tooltip/dist/react-tooltip.css';                                                                                                                                      
                                                                                                                                                                                    
export default function TestPage() {                                                                                                                                                
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;                                                                                                                        
                                                                                                                                                                                    
  // Form state                                                                                                                                                                     
  const [startDate, setStartDate] = useState("2024-10-25");                                                                                                                         
  const [endDate, setEndDate] = useState("2025-02-02");                                                                                                                             
  const [tickerList, setTickerList] = useState("BTCUSDT");                                                                                                                          
  const [dataSource, setDataSource] = useState("binance");                                                                                                                          
  const [timeInterval, setTimeInterval] = useState("30m");                                                                                                                          
  const [technicalIndicators, setTechnicalIndicators] = useState("macd,rsi,cci,dx");                                                                                                
  const [drlLib, setDrlLib] = useState("elegantrl");                                                                                                                                
  const [env, setEnv] = useState("CryptoTradingEnv");                                                                                                                               
  const [modelName, setModelName] = useState("ppo");                                                                                                                                
  const [cwd, setCwd] = useState("papertrading_crypto");                                                                                                                          
  const [initialCapital, setInitialCapital] = useState(10000);                                                                                                                      
  const [netDimension, setNetDimension] = useState("256, 128, 64, 32");                                                                                                             
  const [actorFilename, setActorFilename] = useState("best_actor.pth");                                                                                                                                                                            
  const [atrWindow, setAtrWindow] = useState(14); // Default ATR window                                                                                                             
  const [tpMultiplier, setTpMultiplier] = useState(2); // Default TP multiplier                                                                                                     
  const [slMultiplier, setSlMultiplier] = useState(1); // Default SL multiplier                                                                                                     
  const [maxTradeDuration, setMaxTradeDuration] = useState(20); // Default 20 bars
                                                                                                                                                                                   
  const [response, setResponse] = useState<any>(null);                                                                                                                              
  const [error, setError] = useState<string | null>(null);                                                                                                                          
                                                                                                                                                                                    
  const [logs, setLogs] = useState<string>("");                                                                                                                                     
                                                                                                                                                                                    
  const handleSubmit = async (event: React.FormEvent) => {                                                                                                                          
    event.preventDefault();                                                                                                                                                         
    setError(null);                                                                                                                                                                 
    setResponse(null);                                                                                                                                                              
                                                                                                                                                                                    
    const tickersArray = tickerList.split(",").map((item) => item.trim());                                                                                                          
    const techArray = technicalIndicators.split(",").map((item) => item.trim());                                                                                                    
    const netDimArray = netDimension.split(",").map((s) => Number(s.trim()));                                                                                                       
                                                                                                                                                                                    
    const payload = {                                                                                                                                                               
      start_date: startDate,                                                                                                                                                        
      end_date: endDate,                                                                                                                                                            
      ticker_list: tickersArray,                                                                                                                                                    
      data_source: dataSource,                                                                                                                                                      
      time_interval: timeInterval,                                                                                                                                                  
      technical_indicator_list: techArray,                                                                                                                                          
      drl_lib: drlLib,                                                                                                                                                              
      env: env,                                                                                                                                                                     
      model_name: modelName,                                                                                                                                                        
      cwd: cwd,                                                                                                                                                                     
      initial_capital: initialCapital,                                                                                                                                              
      net_dimension: netDimArray,                                                                                                                                                   
      atr_window: atrWindow,           // Include ATR window                                                                                                                        
      tp_multiplier: tpMultiplier,     // Include TP multiplier                                                                                                                    
      sl_multiplier: slMultiplier,     // Include SL multiplier 
      actor_filename: actorFilename,
      max_trade_duration: maxTradeDuration,                                                                                                                   
    };                                                                                                                                                                              
                                                                                                                                                                                    
    try {                                                                                                                                                                           
      const res = await fetch(`${API_BASE_URL}/test`, {                                                                                                                             
        method: "POST",                                                                                                                                                             
        headers: { "Content-Type": "application/json" },                                                                                                                            
        body: JSON.stringify(payload),                                                                                                                                              
      });                                                                                                                                                                           
      if (!res.ok) {                                                                                                                                                                
        throw new Error(`HTTP error! status: ${res.status}`);                                                                                                                       
      }                                                                                                                                                                             
      const data = await res.json();                                                                                                                                                
      setResponse(data);                                                                                                                                                            
    } catch (err: any) {                                                                                                                                                            
      setError(err.message);                                                                                                                                                        
    }                                                                                                                                                                               
  };                                                                                                                                                                                
                                                                                                                                                                                    
  const fetchLogs = async () => {                                                                                                                                                   
    try {                                                                                                                                                                           
      const res = await fetch(`${API_BASE_URL}/fetch_logs_test`);                                                                                                                   
      if (!res.ok) {                                                                                                                                                                
        throw new Error(`HTTP error! status: ${res.status}`);                                                                                                                       
      }                                                                                                                                                                             
      const data = await res.text();                                                                                                                                                
      setLogs(data);                                                                                                                                                                
    } catch (err: any) {                                                                                                                                                            
      setError(err.message);                                                                                                                                                        
    }                                                                                                                                                                               
  };                                                                                                                                                                                
                                                                                                                                                                                    
  useEffect(() => {                                                                                                                                                                 
    const interval = setInterval(fetchLogs, 6000); // Fetch logs every 6 seconds                                                                                                    
    return () => clearInterval(interval); // Cleanup interval on component unmount                                                                                                  
  }, []);                                                                                                                                                                           
                                                                                                                                                                                    
  return (                                                                                                                                                                          
    <div>                                                                                                                                                                           
      <h1 className="text-2xl font-bold mb-4">Test Your trained Model here!</h1>                                                                                                                        
      <form onSubmit={handleSubmit} className="space-y-6">                                                                                                                          
        <div className="space-y-4">                                                                                                                                                 
          <h2 className="text-lg font-semibold">Test Parameters</h2>                                                                                                                
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">Start Date</label>                                                                                                          
              <input                                                                                                                                                                
                type="date"                                                                                                                                                         
                value={startDate}                                                                                                                                                   
                onChange={(e) => setStartDate(e.target.value)}                                                                                                                      
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">End Date</label>                                                                                                            
              <input                                                                                                                                                                
                type="date"                                                                                                                                                         
                value={endDate}                                                                                                                                                     
                onChange={(e) => setEndDate(e.target.value)}                                                                                                                        
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">Ticker List</label>                                                                                                         
              <input                                                                                                                                                                
                type="text"                                                                                                                                                         
                value={tickerList}                                                                                                                                                  
                onChange={(e) => setTickerList(e.target.value)}                                                                                                                     
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">Data Source</label>                                                                                                         
              <select                                                                                                                                                               
                value={dataSource}                                                                                                                                                  
                onChange={(e) => setDataSource(e.target.value)}                                                                                                                     
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              >                                                                                                                                                                     
                <option value="binance">Binance</option>                                                                                                                            
                <option value="alpaca">Alpaca</option>                                                                                                                              
              </select>                                                                                                                                                             
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">Time Interval</label>                                                                                                       
              <input                                                                                                                                                                
                type="text"                                                                                                                                                         
                value={timeInterval}                                                                                                                                                
                onChange={(e) => setTimeInterval(e.target.value)}                                                                                                                   
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">Technical Indicators</label>                                                                                                
              <input                                                                                                                                                                
                type="text"                                                                                                                                                         
                value={technicalIndicators}                                                                                                                                         
                onChange={(e) => setTechnicalIndicators(e.target.value)}                                                                                                            
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">DRL Library</label>                                                                                                         
              <input                                                                                                                                                                
                type="text"                                                                                                                                                         
                value={drlLib}                                                                                                                                                      
                onChange={(e) => setDrlLib(e.target.value)}                                                                                                                         
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">Environment</label>                                                                                                         
              <input                                                                                                                                                                
                type="text"                                                                                                                                                         
                value={env}                                                                                                                                                         
                onChange={(e) => setEnv(e.target.value)}                                                                                                                            
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">Model Name</label>                                                                                                          
              <input                                                                                                                                                                
                type="text"                                                                                                                                                         
                value={modelName}                                                                                                                                                   
                onChange={(e) => setModelName(e.target.value)}                                                                                                                      
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">CWD</label>                                                                                                                 
              <input                                                                                                                                                                
                type="text"                                                                                                                                                         
                value={cwd}                                                                                                                                                         
                onChange={(e) => setCwd(e.target.value)}                                                                                                                            
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">Initial Capital</label>                                                                                                     
              <input                                                                                                                                                                
                type="number"                                                                                                                                                       
                value={initialCapital}                                                                                                                                              
                onChange={(e) => setInitialCapital(Number(e.target.value))}                                                                                                         
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">Network Dimensions</label>                                                                                                  
              <input                                                                                                                                                                
                type="text"                                                                                                                                                         
                value={netDimension}                                                                                                                                                
                onChange={(e) => setNetDimension(e.target.value)}                                                                                                                   
                required                                                                                                                                                            
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            {/* ATR Window */}                                                                                                                                                       
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">ATR Window</label>                                                                                                          
              <input                                                                                                                                                                
                type="number"                                                                                                                                                       
                value={atrWindow}                                                                                                                                                   
                onChange={(e) => setAtrWindow(Number(e.target.value))}                                                                                                              
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            {/* TP Multiplier */}                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">TP Multiplier</label>                                                                                                       
              <input                                                                                                                                                                
                type="number"                                                                                                                                                       
                value={tpMultiplier}                                                                                                                                                
                onChange={(e) => setTpMultiplier(Number(e.target.value))}                                                                                                           
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>                                                                                                                                                                  
                                                                                                                                                                                    
            {/* SL Multiplier */}                                                                                                                                                    
            <div>                                                                                                                                                                   
              <label className="block font-medium mb-1">SL Multiplier</label>                                                                                                       
              <input                                                                                                                                                                
                type="number"                                                                                                                                                       
                value={slMultiplier}                                                                                                                                                
                onChange={(e) => setSlMultiplier(Number(e.target.value))}                                                                                                           
                className="w-full rounded border border-gray-300 px-3 py-2"                                                                                                         
              />                                                                                                                                                                    
            </div>
            {/* Max Trade Duration */}
            <div>
              <label className="block font-medium mb-1">Max Trade Duration</label>
              <input
                type="number"
                value={maxTradeDuration}
                onChange={(e) => setMaxTradeDuration(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            {/* action filename */}
            <div>
            <label className="block font-medium mb-1">Actor Filename
            <span data-tooltip-id="actor-tooltip" data-tooltip-content="The filename of the actor model to be used for paper trading" className="ml-2 text-blue-500 cursor-pointer">i</span>
            </label>
            <input
              type="text"
              value={actorFilename}
              onChange={(e) => setActorFilename(e.target.value)}
              className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
            />            
            </div>                                                                                                                                                                  
          </div>                                                                                                                                                                    
        </div>                                                                                                                                                                      
                                                                                                                                                                                    
        <div className="mt-6">                                                                                                                                                      
          <button                                                                                                                                                                   
            type="submit"                                                                                                                                                           
            className="px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"                                                               
          >                                                                                                                                                                         
            Start Test                                                                                                                                                              
          </button>                                                                                                                                                                 
        </div>                                                                                                                                                                      
      </form>                                                                                                                                                                       
                                                                                                                                                                                    
      {error && <p className="text-red-500 mt-4">Error: {error}</p>}                                                                                                                
      {response && (                                                                                                                                                                
        <div className="mt-4">                                                                                                                                                      
          <h3 className="font-semibold">Response:</h3>                                                                                                                              
          <pre className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white mt-2">                                               
            {JSON.stringify(response, null, 2)}                                                                                                                                     
          </pre>                                                                                                                                                                    
        </div>                                                                                                                                                                      
      )}                                                                                                                                                                            
      <div className="mt-4">                                                                                                                                                        
        <h3 className="font-semibold">Logs Generated at backend [Refreshes every 60Sec]:</h3>                                                                                       
        <pre className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white mt-2">                                                 
          {logs}                                                                                                                                                                    
        </pre>                                                                                                                                                                      
      </div>                                                                                                                                                                        
    </div>                                                                                                                                                                          
  );                                                                                                                                                                                
}                                                                                                                                                                                   
                  
