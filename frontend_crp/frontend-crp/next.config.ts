import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  /* existing config options here */
  webpack: (config, { dev }) => {
    if (dev) {
      config.devtool = 'source-map';
    }
    return config;
  }
};

export default nextConfig;
