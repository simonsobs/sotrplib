// Vite config to run the lightview dev server for a mock real-time run.
//
// The vite server sends /api/* to lightserve (a proxy). Thus, the browser
// gets the page and the data from one port, and the ssh tunnel needs only
// that port. Run lightview with VITE_SERVICE_URL=/api.
//
// Environment:
//   LIGHTVIEW_DIR      the lightview checkout (with node_modules).
//   LIGHTSERVE_TARGET  the lightserve URL of the services job
//                      (LIGHTSERVE_URL in <run dir>/services.env).
//   LIGHTVIEW_PORT     the port of the vite server. Default: 6979.
//
// Example:
//   cd $LIGHTVIEW_DIR && VITE_SERVICE_URL=/api npx vite \
//       --config <sotrplib>/scripts/mock_real_time/lightview.config.mjs
const dir = process.env.LIGHTVIEW_DIR;
if (!dir || !process.env.LIGHTSERVE_TARGET) {
  throw new Error('Set LIGHTVIEW_DIR and LIGHTSERVE_TARGET.');
}
// The plugin is in the node_modules of lightview, not of this directory.
const { default: react } = await import(
  `${dir}/node_modules/@vitejs/plugin-react/dist/index.js`
);

export default {
  root: dir,
  plugins: [react()],
  server: {
    port: Number(process.env.LIGHTVIEW_PORT || 6979),
    strictPort: true,
    proxy: {
      '/api': {
        target: process.env.LIGHTSERVE_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
};
