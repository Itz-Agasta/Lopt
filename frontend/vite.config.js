import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 5173,
    host: true, // Allow external connections
    cors: true, // Enable CORS
    proxy: {
      // Proxy API calls to backend during development
      '/api': {
        target: process.env.VITE_BACKEND_API || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Environment variables configuration
  envPrefix: 'VITE_', // Only expose VITE_ prefixed variables to the client
})
