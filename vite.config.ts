import { defineConfig } from "vite";

// Relative base so the build works on GitHub Pages (/<repo>/) or any static host.
export default defineConfig({
  base: "./",
  // 5173 is often taken by another project: a fixed port avoids opening the wrong page.
  server: { port: 5190, strictPort: true },
});
