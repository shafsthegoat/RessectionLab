'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');

// Only native file dialogs and the prepared local case directory admit paths.
// The renderer receives opaque list IDs, never authority to open an arbitrary path.
class CaseLibrary {
  constructor(userData, preparedRoot, nativePaths = []) {
    this.record = path.join(userData, 'anatomy-cases.json');
    this.preparedRoot = preparedRoot;
    this.paths = new Map();
    this.nativePaths = nativePaths;
  }
  async read() {
    try {
      const value = JSON.parse(await fs.readFile(this.record, 'utf8'));
      return Array.isArray(value.recent) ? value.recent.filter(p => typeof p === 'string' && path.isAbsolute(p)).slice(0, 12) : [];
    } catch (error) {
      if (error.code === 'ENOENT' || error instanceof SyntaxError) return [];
      throw error;
    }
  }
  async remember(filename) {
    const resolved = await fs.realpath(filename);
    const recent = [resolved, ...(await this.read()).filter(p => p !== resolved)].slice(0, 12);
    await fs.mkdir(path.dirname(this.record), {recursive: true});
    const temporary = this.record + '.tmp';
    await fs.writeFile(temporary, JSON.stringify({recent}, null, 2), {mode: 0o600});
    await fs.rename(temporary, this.record);
  }
  async list() {
    const recent = await this.read();
    let names = [];
    try { names = await fs.readdir(this.preparedRoot); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const prepared = names.filter(n => n.endsWith('.ressectionlab')).sort().map(n => path.join(this.preparedRoot, n));
    const rows = [];
    this.paths.clear();
    for (const filename of [...new Set([...recent, ...this.nativePaths, ...prepared])]) {
      try {
        const stat = await fs.stat(filename);
        if (!stat.isFile()) continue;
        const id = require('node:crypto').createHash('sha256').update(filename).digest('hex');
        const native=path.basename(filename)==='patient-anatomy.json';
        const doc=native?JSON.parse(await fs.readFile(filename,'utf8')):null;
        if(native&&doc.schema!=='patient-anatomy-v1')continue;
        this.paths.set(id, filename);
        rows.push({id, title: native?doc.patient_id:path.basename(filename, '.ressectionlab'), recent: recent.includes(filename), bytes: stat.size, ...(native?{native:true}:{})});
      } catch (error) { if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error; }
    }
    return rows;
  }
  async resolve(id) {
    await this.list();
    if (typeof id !== 'string' || !this.paths.has(id)) throw new Error('Select a saved case from the current library.');
    return this.paths.get(id);
  }
}
module.exports = {CaseLibrary};
