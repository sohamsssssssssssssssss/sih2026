import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  devIndicators: false,
  async rewrites() {
    return [
      { source: "/console", destination: "/reference/index.html" },
      { source: "/console/:path*", destination: "/reference/:path*" },
    ];
  },
};

export default nextConfig;
