import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In development the page is served by Vite and /api is forwarded to the backend.
// Attune uses its own ports (page 5180, backend 8787) so it can run beside other projects.
const backend = `http://localhost:${process.env.ATTUNE_BACKEND_PORT ?? 8787}`
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5180, strictPort: true, proxy: { '/api': { target: backend, changeOrigin: true } } },
})
