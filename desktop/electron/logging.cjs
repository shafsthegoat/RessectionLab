'use strict';

const fs = require('node:fs');
const path = require('node:path');

/** Local-only diagnostics survive Finder launches, where stderr is invisible. */
function createLogger(directory, { maxBytes = 5 * 1024 * 1024, stderr = process.stderr } = {}) {
  fs.mkdirSync(directory, { recursive: true, mode: 0o700 });
  const filename = path.join(directory, 'desktop.log');
  return message => {
    const record = `${new Date().toISOString()} ${String(message).slice(0, 16000).trim()}\n`;
    try {
      // Rotate during long sessions too; keep only one prior diagnostic file.
      if (fs.existsSync(filename) && fs.statSync(filename).size + Buffer.byteLength(record) > maxBytes) {
        fs.renameSync(filename, path.join(directory, 'desktop.previous.log'));
      }
      fs.appendFileSync(filename, record, { mode: 0o600 });
    } catch { /* Diagnostics must not interrupt image work. */ }
    stderr.write(record);
  };
}

module.exports = { createLogger };
