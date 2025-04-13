"use client";
import { useState } from "react";
import { Tooltip } from "react-tooltip";
import 'react-tooltip/dist/react-tooltip.css';

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

  const [logFilePath, setLogFilePath] = useState("/home/souz_wsl/finrl_proj/FinRL_Meta/api/papertrading_crypto/training.log");
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
      log_file_path: logFilePath,  // Use logFilePath from state
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
          <label className="block font-medium mb-1">
            Start Date
            <span data-tooltip-id="start-date-tooltip" data-tooltip-content="Format: YYYY-MM-DD" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="start-date-tooltip" />
          </label>
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
          <label className="block font-medium mb-1">
            End Date
            <span data-tooltip-id="end-date-tooltip" data-tooltip-content="Format: YYYY-MM-DD" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="end-date-tooltip" />
          </label>
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
          <label className="block font-medium mb-1">
            Ticker List (comma-separated)
            <span data-tooltip-id="ticker-tooltip" data-tooltip-content="List of tickers, e.g., BTCUSDT,ETHUSDT" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="ticker-tooltip" />
          </label>
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
          <label className="block font-medium mb-1">
            Data Source
            <span data-tooltip-id="data-source-tooltip" data-tooltip-content="Source of data, e.g., binance" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="data-source-tooltip" />
          </label>
          <input
            type="text"
            value={dataSource}
            onChange={(e) => setDataSource(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Time Interval */}
        <div>
          <label className="block font-medium mb-1">
            Time Interval
            <span data-tooltip-id="time-interval-tooltip" data-tooltip-content="Interval between data points, e.g., 15Min" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="time-interval-tooltip" />
          </label>
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
          <label className="block font-medium mb-1">
            DRL Lib
            <span data-tooltip-id="drl-lib-tooltip" data-tooltip-content="Deep Reinforcement Learning library, e.g., elegantrl" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="drl-lib-tooltip" />
          </label>
          <input
            type="text"
            value={drlLib}
            onChange={(e) => setDrlLib(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Env */}
        <div>
          <label className="block font-medium mb-1">
            Env
            <span data-tooltip-id="env-tooltip" data-tooltip-content="Environment class, e.g., CryptoTradingEnv" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="env-tooltip" />
          </label>
          <input
            type="text"
            value={env}
            onChange={(e) => setEnv(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Model Name */}
        <div>
          <label className="block font-medium mb-1">
            Model Name
            <span data-tooltip-id="model-name-tooltip" data-tooltip-content="Name of the model, e.g., ppo" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="model-name-tooltip" />
          </label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* CWD */}
        <div>
          <label className="block font-medium mb-1">
            CWD
            <span data-tooltip-id="cwd-tooltip" data-tooltip-content="Current Working Directory. Please don't change it if you are not developer!" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="cwd-tooltip" />
          </label>
          <input
            type="text"
            value={cwd}
            onChange={(e) => setCwd(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Break Step */}
        <div>
          <label className="block font-medium mb-1">
            Break Step
            <span data-tooltip-id="break-step-tooltip" data-tooltip-content="Number of steps after which training should stop, e.g., 100000" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="break-step-tooltip" />
          </label>
          <input
            type="number"
            value={breakStep}
            onChange={(e) => setBreakStep(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2"
          />
        </div>

        {/* Log File Path */}
        <div>
          <label className="block font-medium mb-1">
            Log File Path
            <span data-tooltip-id="log-file-path-tooltip" data-tooltip-content="Path to the log file, e.g., /home/souz_wsl/finrl_proj/FinRL_Meta/api/papertrading_crypto/training.log" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="log-file-path-tooltip" />
          </label>
          <input
            type="text"
            value={logFilePath}
            onChange={(e) => setLogFilePath(e.target.value)}
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
