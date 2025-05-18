import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,

  // Remove webpack config — Turbopack doesn't use it
  // If you need source maps, see below
};

export default nextConfig;


