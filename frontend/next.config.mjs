/** @type {import('next').NextConfig} */
const nextConfig = {
  // Lets a production build use its own folder (NEXT_DIST_DIR=.next-prod) without
  // disturbing a dev server that is running from .next.
  distDir: process.env.NEXT_DIST_DIR || ".next",
  poweredByHeader: false,
  compress: true,
  experimental: {
    optimizePackageImports: ["lucide-react", "recharts", "reactflow"],
    // The /api rewrite proxies to FastAPI. Next's default proxy timeout is 30 s, but a local
    // model can take longer (filing search + general-knowledge fallback), which surfaced as a
    // spurious "HTTP 500". Allow up to 5 minutes.
    proxyTimeout: 300_000,
  },
  images: {
    // Serve AVIF/WebP when the browser supports them; cache optimized images for a week.
    formats: ["image/avif", "image/webp"],
    minimumCacheTTL: 60 * 60 * 24 * 7,
  },
  async headers() {
    return [
      {
        // Brand artwork changes rarely: let browsers cache it for a week.
        source: "/brand/:path*",
        headers: [{ key: "Cache-Control", value: "public, max-age=604800, stale-while-revalidate=86400" }],
      },
    ];
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
    ];
  },
};

export default nextConfig;
