import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  server: {
    allowedHosts: [
      'ec2-3-111-30-241.ap-south-1.compute.amazonaws.com'
    ]
  },
  plugins: [react(),
    tailwindcss(),
  ],
})
