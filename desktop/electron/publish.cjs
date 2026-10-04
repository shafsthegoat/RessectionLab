'use strict';

const filesystem = require('node:fs/promises');
const path = require('node:path');
const { randomUUID } = require('node:crypto');

/** Publish only a signed, verified staging directory; retain and restore the prior app. */
async function publishBundle(stagedDirectory, finalDirectory, fs = filesystem) {
  let previousDirectory;
  try {
    await fs.access(finalDirectory);
    const candidate = path.join(path.dirname(finalDirectory), `previous-${Date.now()}-${randomUUID().slice(0, 8)}`);
    await fs.rename(finalDirectory, candidate);
    previousDirectory = candidate;
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  try {
    await fs.rename(stagedDirectory, finalDirectory);
  } catch (error) {
    if (previousDirectory) {
      try { await fs.rename(previousDirectory, finalDirectory); }
      catch (restoreError) {
        throw new Error(`Publishing failed; the prior app remains at ${previousDirectory}. Restore failed: ${restoreError.message}`, { cause: error });
      }
    }
    throw error;
  }
  return { previousDirectory };
}

module.exports = { publishBundle };
