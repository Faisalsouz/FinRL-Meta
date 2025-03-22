"use client";
import { useState } from "react";

export default function TrainPage() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

  // Form state
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [tickerList, setTickerList] = useState("");
  const [dataSource, setDataSource] = useState("binance");
  const [timeInterval, setTimeInterval] = useState("15Min");
  const [technicalIndicators, setTechnicalIndicators] = useState("");
  const [drlLib, setDrlLib] = useState("elegantrl");
  const [env, setEnv] = useState("CryptoTradingEnv");
  const [modelName, setModelName] = useState("ppo");
  const [cwd, setCwd] = useState("./papertrading_crypto");
  const [breakStep, setBreakStep] = useState(100000);

  const [response, setResponse] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);
    setResponse(null);

    const tickersArray = tickerList.split(",").map((item) => item.trim());
    const techArray = technicalIndicators.split(",").map((item) => item.trim());

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
      break_step: Number(breakStep),
    };

    try {
      const res = await fetch(`${API_BASE_URL}/train`, {
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
      <h1 className="text-2xl font-bold mb-4">Training Form</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Start Date */}
        <div>
          <label className="block font-medium mb-1">Start Date</label>
          <input
            type="text"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            placeholder="YYYY-MM-DD"
            required
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* End Date */}
        <div>
          <label className="block font-medium mb-1">End Date</label>
          <input
            type="text"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            placeholder="YYYY-MM-DD"
            required
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Ticker List */}
        <div>
          <label className="block font-medium mb-1">Ticker List (comma-separated)</label>
          <input
            type="text"
            value={tickerList}
            onChange={(e) => setTickerList(e.target.value)}
            placeholder="BTCUSDT,ETHUSDT"
            required
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Data Source */}
        <div>
          <label className="block font-medium mb-1">Data Source</label>
          <input
            type="text"
            value={dataSource}
            onChange={(e) => setDataSource(e.target.value)}
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

        {/* Technical Indicators */}
        <div>
          <label className="block font-medium mb-1">
            Technical Indicators (comma-separated)
          </label>
          <input
            type="text"
            value={technicalIndicators}
            onChange={(e) => setTechnicalIndicators(e.target.value)}
            placeholder="macd,rsi,cci"
            required
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

        {/* Env */}
        <div>
          <label className="block font-medium mb-1">Env</label>
          <input
            type="text"
            value={env}
            onChange={(e) => setEnv(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Model Name */}
        <div>
          <label className="block font-medium mb-1">Model Name</label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
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

        {/* Break Step */}
        <div>
          <label className="block font-medium mb-1">Break Step</label>
          <input
            type="number"
            value={breakStep}
            onChange={(e) => setBreakStep(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        <button
          type="submit"
          className="px-4 py-2 bg-blue-600 text-white font-semibold rounded hover:bg-blue-700"
        >
          Start Training
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
    </div>
  );
}
