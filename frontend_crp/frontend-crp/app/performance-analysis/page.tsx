"use client";

import React, { useState, useEffect } from "react";
import axios from 'axios';

export default function PerformanceAnalysisPage() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;
  const [performanceData, setPerformanceData] = useState(null);
  const [startDate, setStartDate] = useState("2025-01-01");
  const [endDate, setEndDate] = useState("2025-02-01");
  const [error, setError] = useState(null);


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
      {error && <div>Error: {error}</div>}
      {!performanceData ? (
        <div>Loading...</div>
      ) : (
        <table className="min-w-full bg-white dark:bg-gray-800">
          <thead>
            <tr>
              <th className="py-2">Date</th>
              <th className="py-2">ATR Average</th>
              <th className="py-2">Profit/Loss</th>
              <th className="py-2">Equity Change</th>
              <th className="py-2">Average Trade Per Day</th>
              <th className="py-2">Winning Trades</th>
              <th className="py-2">Lost Trades</th>
              <th className="py-2">Timeout Trades</th>
              <th className="py-2">Average Win</th>
              <th className="py-2">Average Loss</th>
              <th className="py-2">Stop Loss Stats</th>
            </tr>
          </thead>
          <tbody>
            {performanceData && performanceData.dates && performanceData.dates.map((date, index) => (
              <tr key={index}>
                <td className="border px-4 py-2">{date}</td>
                <td className="border px-4 py-2">{performanceData.atr_avg[index]}</td>
                <td className="border px-4 py-2">{performanceData.profit_loss[index]}</td>
                <td className="border px-4 py-2">{performanceData.equity_change[index]}</td>
                <td className="border px-4 py-2">{performanceData.avg_trade_per_day[index]}</td>
                <td className="border px-4 py-2">{performanceData.win_trades[index]}</td>
                <td className="border px-4 py-2">{performanceData.lost_trades[index]}</td>
                <td className="border px-4 py-2">{performanceData.timeout_trades[index]}</td>
                <td className="border px-4 py-2">{performanceData.avg_win[index]}</td>
                <td className="border px-4 py-2">{performanceData.avg_loss[index]}</td>
                <td className="border px-4 py-2">{performanceData.stop_loss_stats[index]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
