"use client";
import React, { useState, useEffect } from "react";
import axios from 'axios';
import { Tooltip } from "react-tooltip";
import 'react-tooltip/dist/react-tooltip.css';

export default function PaperTradingPage() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

  // Form state
  const [tickerList, setTickerList] = useState("BTC/USD");
  const [timeInterval, setTimeInterval] = useState("5min");
  const [drlLib, setDrlLib] = useState("elegantrl");
  const [agent, setAgent] = useState("ppo");
  const [cwd, setCwd] = useState("papertrading_crypto");
  const [netDim, setNetDim] = useState("256, 128, 64, 32");
  const [stateDim, setStateDim] = useState(7);
  const [actionDim, setActionDim] = useState(1);
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [apiBaseUrl, setApiBaseUrl] = useState("https://paper-api.alpaca.markets");
  const [techIndicatorList, setTechIndicatorList] = useState("macd,rsi,cci,dx");  
  const [maxStock, setMaxStock] = useState(100.0);
  const [tpMultiplier, setTpMultiplier] = useState(2);
  const [slMultiplier, setSlMultiplier] = useState(5);
  const [atrWindow, setAtrWindow] = useState(14);
  const [maxTradeDuration, setMaxTradeDuration] = useState(20);
  const [logFilePath, setLogFilePath] = useState("/home/souz_wsl/finrl_proj/FinRL_Meta/api/papertrading_crypto/paper_trading.log");

  // Response & error
  const [response, setResponse] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const [logs, setLogs] = useState<string>("");

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    // Convert comma-separated strings
    const tickersArray = tickerList.split(",").map((s) => s.trim());
    const netDimArray = netDim.split(",").map((s) => Number(s.trim()));
    const techArray = techIndicatorList.split(",").map((s) => s.trim());

    const payload = {
      ticker_list: tickersArray,
      time_interval: timeInterval,
      drl_lib: drlLib,
      agent: agent,
      cwd: cwd,
      net_dim: netDimArray,
      state_dim: Number(stateDim),
      action_dim: Number(actionDim),
      API_KEY: apiKey,
      API_SECRET: apiSecret,
      API_BASE_URL: apiBaseUrl,
      tech_indicator_list: techArray,
      max_stock: Number(maxStock),
      tp_multiplier: tpMultiplier,
      sl_multiplier: slMultiplier,
      atr_window: atrWindow,
      max_trade_duration: maxTradeDuration,
      log_file_path: logFilePath,
    };
    console.log("The payload :", payload);

    try {
      const res = await fetch(`${API_BASE_URL}/paper_trade`, {
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

  const handleStopPaperTrading = async () => {
    setResponse(null);
    setError(null);

    try {
      const res = await fetch(`${API_BASE_URL}/stop_paper_trade`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      const res = await fetch(`${API_BASE_URL}/fetch_logs`);
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
      <h1 className="text-2xl font-bold mb-4">Paper Trading Form</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Ticker List */}
        <div>
          <label className="block font-medium mb-1">
            Ticker List (comma-separated)
            <span data-tooltip-id="ticker-tooltip" data-tooltip-content="Currently supports only single asset.i.e format BTC/USD" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="ticker-tooltip" />
          </label>
          <input
            type="text"
            value={tickerList}
            onChange={(e) => setTickerList(e.target.value)}
            placeholder="BTC/USD,ETH/USD,SOL/USD"
            required
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* Time Interval */}
        <div>
          <label className="block font-medium mb-1">Time Interval
          <span data-tooltip-id="time-tooltip" data-tooltip-content="the format must be in small 'min'i.e. 5min, 10min" className="ml-2 text-blue-500 cursor-pointer">i</span>
            </label>
          <input
            type="text"
            value={timeInterval}
            onChange={(e) => setTimeInterval(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* DRL Lib */}
        <div>
          <label className="block font-medium mb-1">DRL Lib</label>
          <input
            type="text"
            value={drlLib}
            onChange={(e) => setDrlLib(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* Agent */}
        <div>
          <label className="block font-medium mb-1">
            Agent
            <span data-tooltip-id="agent-tooltip" data-tooltip-content="Currently supports only PPO type" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="agent-tooltip" />
          </label>
          <input
            type="text"
            value={agent}
            onChange={(e) => setAgent(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* CWD */}
        <div>
          <label className="block font-medium mb-1">CWD
          <span data-tooltip-id="cwd-tooltip" data-tooltip-content="Current Working Directory. Please don't change it if you are not developer!" className="ml-2 text-blue-500 cursor-pointer">i</span>
          </label>
          <input
            type="text"
            value={cwd}
            onChange={(e) => setCwd(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* Net Dim */}
        <div>
          <label className="block font-medium mb-1">Net Dim (comma-separated)
          <span data-tooltip-id="net-tooltip" data-tooltip-content="How Dense network should be. this value must match with value you have 
          set in training">i</span>
          <Tooltip id="net-tooltip" />
          </label>
          <input
            type="text"
            value={netDim}
            onChange={(e) => setNetDim(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* State Dim */}
        <div>
          <label className="block font-medium mb-1">
            State Dim
            <span data-tooltip-id="state-tooltip" data-tooltip-content="Calculated as: 1 (scaled price) + len(INDICATORS) (technical indicators) + 1 (previous step’s percentage change in price) + 1 (in-trade flag)" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="state-tooltip" />
          </label>
          <input
            type="number"
            value={stateDim}
            onChange={(e) => setStateDim(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* Action Dim */}
        <div>
          <label className="block font-medium mb-1">
            Action Dim
            <span data-tooltip-id="action-tooltip" data-tooltip-content="Always equal to the number of assets, currently supports only single asset, so it should be 1" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="action-tooltip" />
          </label>
          <input
            type="number"
            value={actionDim}
            onChange={(e) => setActionDim(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* API Key */}
        <div>
          <label className="block font-medium mb-1">API Key</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            required
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* API Secret */}
        <div>
          <label className="block font-medium mb-1">API Secret</label>
          <input
            type="password"
            value={apiSecret}
            onChange={(e) => setApiSecret(e.target.value)}
            required
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* API Base URL */}
        <div>
          <label className="block font-medium mb-1">API Base URL</label>
          <input
            type="text"
            value={apiBaseUrl}
            onChange={(e) => setApiBaseUrl(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* Tech Indicator List */}
        <div>
          <label className="block font-medium mb-1">
            Tech Indicator List (comma-separated)
            <span data-tooltip-id="tech-tooltip" data-tooltip-content="List of technical indicators, e.g., macd,rsi,cci,dx" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="tech-tooltip" />
          </label>
          <input
            type="text"
            value={techIndicatorList}
            onChange={(e) => setTechIndicatorList(e.target.value)}
            placeholder="macd,rsi,cci,dx"
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        {/* Max Stock */}
        <div>
          <label className="block font-medium mb-1">Max Stock</label>
          <input
            type="number"
            value={maxStock}
            onChange={(e) => setMaxStock(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>

        <div>
          <label className="block font-medium mb-1">TP Multiplier</label>
          <input
            type="number"
            step="0.1"
            value={tpMultiplier}
            onChange={(e) => setTpMultiplier(Number(e.target.value))}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        <div>
          <label className="block font-medium mb-1">SL Multiplier</label>
          <input
            type="number"
            step="0.1"
            value={slMultiplier}
            onChange={(e) => setSlMultiplier(Number(e.target.value))}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        <div>
          <label className="block font-medium mb-1">ATR Window</label>
          <input
            type="number"
            value={atrWindow}
            onChange={(e) => setAtrWindow(Number(e.target.value))}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        <div>
          <label className="block font-medium mb-1">Max Trade Duration</label>
          <input
            type="number"
            value={maxTradeDuration}
            onChange={(e) => setMaxTradeDuration(Number(e.target.value))}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        <button
          type="button"
          onClick={handleStopPaperTrading}
          className="px-4 py-2 bg-red-600 text-white font-semibold rounded hover:bg-red-700 ml-4"
        >
          Stop Paper Trading
        </button>
        <button
          type="submit"
          className="px-4 py-2 bg-green-600 text-white font-semibold rounded hover:bg-green-700"
        >
          Start Paper Trading
        </button>
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
