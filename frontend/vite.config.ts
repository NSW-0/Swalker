import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // The FastAPI backend (backend/), run with: python -m uvicorn app.main:create_app --factory --port 8000
    proxy: { '/api': 'http://localhost:8000' },
  },
})
