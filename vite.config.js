import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    https: false,
    host: true,
  },
  optimizeDeps: {
    exclude: ['@mediapipe/tasks-vision'],
  },
});
