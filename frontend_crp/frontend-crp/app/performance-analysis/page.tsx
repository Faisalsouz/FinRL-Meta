import { useState, useEffect } from 'react';
import axios from 'axios';

export default function PerformanceAnalysisPage() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;
  const [performanceData, setPerformanceData] = useState(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  useEffect(() => {
    fetchPerformanceData();
  }, []);

  const fetchPerformanceData = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/fetch_performance_analysis`, {
        params: { start_date: startDate, end_date: endDate }
      });
      setPerformanceData(response.data);
    } catch (error) {
      console.error("Error fetching performance data:", error);
    }
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Performance Analysis</h1>
      <form onSubmit={(e) => { e.preventDefault(); fetchPerformanceData(); }} className="space-y-4">
        <div>
          <label className="block font-medium mb-1">Start Date</label>
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>
        <div>
          <label className="block font-medium mb-1">End Date</label>
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="w-full rounded border border-gray-300 px-3 py-2 bg-white dark:bg-gray-800 text-black dark:text-white"
          />
        </div>
        <button type="submit" className="px-4 py-2 bg-blue-500 text-white rounded">Fetch Data</button>
      </form>
      {performanceData && (
        <div className="mt-4">
          <h2 className="text-xl font-bold">Performance Metrics</h2>
          <p>ATR Average: {performanceData.atr_avg}</p>
          <p>Profit and Loss: {JSON.stringify(performanceData.profit_loss)}</p>
          <p>Change in Equity: {performanceData.equity_change}</p>
          <p>Average Trade per Day: {performanceData.avg_trade_per_day}</p>
          <p>Win Trades: {performanceData.win_trades}</p>
          <p>Lost Trades: {performanceData.lost_trades}</p>
          <p>Timeout Trades: {performanceData.timeout_trades}</p>
          <p>Average Win: {performanceData.avg_win}</p>
          <p>Average Loss: {performanceData.avg_loss}</p>
          <p>Stop Loss Stats: {performanceData.stop_loss_stats}</p>
        </div>
      )}
    </div>
  );
}
