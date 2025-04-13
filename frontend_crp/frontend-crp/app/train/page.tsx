"use client";
import { useState, useEffect } from "react";
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

  // ERL Parameters state
  const [learningRate, setLearningRate] = useState(3e-6);
  const [batchSize, setBatchSize] = useState(512);
  const [gamma, setGamma] = useState(0.99);
  const [seed, setSeed] = useState(312);
  const [netDimension, setNetDimension] = useState("256, 128, 64, 32");
  const [targetStep, setTargetStep] = useState(5000);
  const [evalGap, setEvalGap] = useState(30);
  const [evalTimes, setEvalTimes] = useState(1);
  const [ratioClip, setRatioClip] = useState(0.5);
  const [lambdaGaeAdv, setLambdaGaeAdv] = useState(0.95);
  const [lambdaEntropy, setLambdaEntropy] = useState(0.01);

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

    const erlParams = {
      learning_rate: learningRate,
      batch_size: batchSize,
      gamma: gamma,
      seed: seed,
      net_dimension: netDimArray,
      target_step: targetStep,
      eval_gap: evalGap,
      eval_times: evalTimes,
      ratio_clip: ratioClip,
      lambda_gae_adv: lambdaGaeAdv,
      lambda_entropy: lambdaEntropy,
    };

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
      erl_params: erlParams,  // Include ERL parameters
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

  const fetchLogs = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/fetch_logs_train`);
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
            <span data-tooltip-id="ticker-tooltip" data-tooltip-content="Right now we only support single asset" className="ml-2 text-blue-500 cursor-pointer">i</span>
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
            <span data-tooltip-id="technical-indicators-tooltip" data-tooltip-content="List of technical indicators, e.g., macd,rsi,cci" className="ml-2 text-blue-500 cursor-pointer">i</span>
            <Tooltip id="technical-indicators-tooltip" />
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


        {/* ERL Parameters Subform */}
        <div className="mt-4">
          <h2 className="text-xl font-bold mb-2">ERL Parameters</h2>
          <div className="space-y-4">
            <div>
              <label className="block font-medium mb-1">Learning Rate</label>
              <input
                type="number"
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Batch Size</label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Gamma</label>
              <input
                type="number"
                value={gamma}
                onChange={(e) => setGamma(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Seed</label>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Net Dimension (comma-separated)</label>
              <input
                type="text"
                value={netDimension}
                onChange={(e) => setNetDimension(e.target.value)}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Target Step</label>
              <input
                type="number"
                value={targetStep}
                onChange={(e) => setTargetStep(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Eval Gap</label>
              <input
                type="number"
                value={evalGap}
                onChange={(e) => setEvalGap(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Eval Times</label>
              <input
                type="number"
                value={evalTimes}
                onChange={(e) => setEvalTimes(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Ratio Clip</label>
              <input
                type="number"
                value={ratioClip}
                onChange={(e) => setRatioClip(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Lambda GAE Adv</label>
              <input
                type="number"
                value={lambdaGaeAdv}
                onChange={(e) => setLambdaGaeAdv(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
            <div>
              <label className="block font-medium mb-1">Lambda Entropy</label>
              <input
                type="number"
                value={lambdaEntropy}
                onChange={(e) => setLambdaEntropy(Number(e.target.value))}
                className="w-full rounded border border-gray-300 px-3 py-2"
              />
            </div>
          </div>
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
      <div className="mt-4">
        <h3 className="font-semibold">Logs Generated at backend [Refreshes every 60Sec]:</h3>
        <pre className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white mt-2">
          {logs}
        </pre>
      </div>
    </div>
  );
}
