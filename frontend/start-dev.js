// Helper to start vite from the frontend directory
process.chdir(__dirname);
require('child_process').execFileSync(
  process.execPath,
  ['node_modules/vite/bin/vite.js', '--port', '3000'],
  { stdio: 'inherit' }
);
