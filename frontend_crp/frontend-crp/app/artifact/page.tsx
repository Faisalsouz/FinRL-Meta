"use client";
import { useState, useEffect } from "react";
import { Tooltip } from "react-tooltip";
import 'react-tooltip/dist/react-tooltip.css';

export default function ArtifactPage() {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL;

  const [artifacts, setArtifacts] = useState([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchArtifacts = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/list_artifacts`);
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        const data = await res.json();
        setArtifacts(data.files);
      } catch (err: any) {
        setError(err.message);
      }
    };

    fetchArtifacts();
  }, []);

  return (
    <div className="mt-4">
      <h1 className="text-3xl font-bold mb-6">Artifacts</h1>
      {error && <p className="text-red-500 mt-4">Error: {error}</p>}
      <ul className="space-y-2 list-none">
        {artifacts.map((artifact: any) => (
          <li key={artifact.name} className="border rounded p-4">
            <div className="flex justify-between items-center">
              <div>
                <p className="font-semibold">{artifact.name}</p>
                <p>Created At: {new Date(artifact.created_at * 1000).toLocaleString()}</p>
                <p>Modified At: {new Date(artifact.modified_at * 1000).toLocaleString()}</p>
              </div>
              <a
                href={`${API_BASE_URL}/download_artifact?path=${encodeURIComponent(artifact.path)}`}
                className="text-blue-600 hover:underline"
                download
              >
                Download
              </a>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
