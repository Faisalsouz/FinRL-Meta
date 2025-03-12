"use client";
import { useState } from "react";

export default function PaperTradingPage() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

  // Form state
  const [tickerList, setTickerList] = useState("");
  const [timeInterval, setTimeInterval] = useState("5Min");
  const [drlLib, setDrlLib] = useState("elegantrl");
  const [agent, setAgent] = useState("ppo");
  const [cwd, setCwd] = useState("./papertrading_crypto");
  const [netDim, setNetDim] = useState("128,64");
  const [stateDim, setStateDim] = useState(22);
  const [actionDim, setActionDim] = useState(3);
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [apiBaseUrl, setApiBaseUrl] = useState("https://paper-api.alpaca.markets");
  const [techIndicatorList, setTechIndicatorList] = useState("");
  const [maxStock, setMaxStock] = useState(100.0);

  // Response & error
  const [response, setResponse] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setResponse(null);
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
    };

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

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Paper Trading Form</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Ticker List */}
        <div>
          <label className="block font-medium mb-1">
            Ticker List (comma-separated)
          </label>
          <input
            type="text"
            value={tickerList}
            onChange={(e) => setTickerList(e.target.value)}
            placeholder="BTCUSD,ETHUSD,SOLUSD"
            required
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Time Interval */}
        <div>
          <label className="block font-medium mb-1">Time Interval</label>
          <input
            type="text"
            value={timeInterval}
            onChange={(e) => setTimeInterval(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* DRL Lib */}
        <div>
          <label className="block font-medium mb-1">DRL Lib</label>
          <input
            type="text"
            value={drlLib}
            onChange={(e) => setDrlLib(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Agent */}
        <div>
          <label className="block font-medium mb-1">Agent</label>
          <input
            type="text"
            value={agent}
            onChange={(e) => setAgent(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* CWD */}
        <div>
          <label className="block font-medium mb-1">CWD</label>
          <input
            type="text"
            value={cwd}
            onChange={(e) => setCwd(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Net Dim */}
        <div>
          <label className="block font-medium mb-1">Net Dim (comma-separated)</label>
          <input
            type="text"
            value={netDim}
            onChange={(e) => setNetDim(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* State Dim */}
        <div>
          <label className="block font-medium mb-1">State Dim</label>
          <input
            type="number"
            value={stateDim}
            onChange={(e) => setStateDim(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Action Dim */}
        <div>
          <label className="block font-medium mb-1">Action Dim</label>
          <input
            type="number"
            value={actionDim}
            onChange={(e) => setActionDim(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* API Key */}
        <div>
          <label className="block font-medium mb-1">API Key</label>
          <input
            type="text"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            required
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* API Secret */}
        <div>
          <label className="block font-medium mb-1">API Secret</label>
          <input
            type="text"
            value={apiSecret}
            onChange={(e) => setApiSecret(e.target.value)}
            required
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* API Base URL */}
        <div>
          <label className="block font-medium mb-1">API Base URL</label>
          <input
            type="text"
            value={apiBaseUrl}
            onChange={(e) => setApiBaseUrl(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Tech Indicator List */}
        <div>
          <label className="block font-medium mb-1">
            Tech Indicator List (comma-separated)
          </label>
          <input
            type="text"
            value={techIndicatorList}
            onChange={(e) => setTechIndicatorList(e.target.value)}
            placeholder="macd,rsi,cci,dx"
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Max Stock */}
        <div>
          <label className="block font-medium mb-1">Max Stock</label>
          <input
            type="number"
            value={maxStock}
            onChange={(e) => setMaxStock(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

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
          <pre className="bg-gray-100 p-4 rounded mt-2">
            {JSON.stringify(response, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
