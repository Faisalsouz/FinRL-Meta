"use client"; // If you plan to use React hooks here. Otherwise, you can omit it if purely static.

import Link from "next/link";

export default function HomePage() {
  return (
    <div className="mt-4">
      <h1 className="text-3xl font-bold mb-6">Welcome to CryptoMatic with AI</h1>
      <p className="mb-6">
        This application allows you to train a DRL model, test it, and run paper trading.
      </p>
      <ul className="space-y-2 list-none">
        <li>
          <Link href="/train" className="text-blue-600 hover:underline">
            Go to Training Form
          </Link>
        </li>
        <li>
          <Link href="/test" className="text-blue-600 hover:underline">
            Go to Testing Form
          </Link>
        </li>
        <li>
          <Link href="/paper-trading" className="text-blue-600 hover:underline">
            Go to Paper Trading Form
          </Link>
        </li>
        <li>
          <Link href="/performance-analysis" className="text-blue-600 hover:underline">
            Go to Performance Analysis
          </Link>
        </li>
      </ul>
    </div>
  );
}
