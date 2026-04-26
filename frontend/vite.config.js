import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Override with: OPENENV_BACKEND_URL=http://localhost:8000 npm run dev
// Defaults to the standard OpenEnv server port.
const backend = process.env.OPENENV_BACKEND_URL ?? 'http://localhost:8000'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/',
  server: {
    proxy: {
      '/api': backend,
      '/reset': backend,
      '/step': backend,
      '/state': backend,
    }
  }
})
