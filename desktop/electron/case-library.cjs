'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const {PatientCatalog} = require('./patient-catalog.cjs');

async function readMetadata(filename, maxBytes) {
  const file=await fs.open(filename,'r');
  try {
    const stat=await file.stat();
    if (!stat.isFile() || stat.size>maxBytes) throw new Error('Saved case metadata exceeds the library read limit.');
    const buffer=Buffer.alloc(stat.size+1),{bytesRead}=await file.read(buffer,0,buffer.length,0);
    if(bytesRead!==stat.size)throw new Error('Saved case metadata changed during listing.');
    return JSON.parse(buffer.subarray(0,bytesRead).toString('utf8'));
  } finally {await file.close();}
}

// Native file dialogs, prepared cases and the fixed source-checked catalog admit paths.
// The renderer receives opaque list IDs, never authority to open an arbitrary path.
class CaseLibrary {
  constructor(userData, preparedRoot, nativePaths = [], catalogRoot = null) {
    this.record = path.join(userData, 'anatomy-cases.json');
    this.preparedRoot = preparedRoot;
    this.paths = new Map();
    this.nativePaths = nativePaths;
    this.catalog = catalogRoot ? new PatientCatalog(catalogRoot) : null;
  }
  async read() {
    try {
      const value = await readMetadata(this.record,64*1024);
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
    // Check catalog metadata before touching any listed bundle. Listing never
    // reads ZIP/image payloads or accepts a catalog path from the renderer.
    const catalog = this.catalog ? await this.catalog.load() : [];
    const catalogAliases = new Map(catalog.flatMap(row => row.aliases.map(filename => [filename,row])));
    const recent = await this.read();
    let names = [];
    try { names = await fs.readdir(this.preparedRoot); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    const prepared = names.filter(n => n.endsWith('.ressectionlab')).sort().map(n => path.join(this.preparedRoot, n));
    const rows = [];
    this.paths.clear();
    const addedCatalog = new Set();
    const addCatalog = (row,wasRecent=false) => {
      if (addedCatalog.has(row.id)) return;
      addedCatalog.add(row.id);this.paths.set(row.id,{catalog:true});
      rows.push({id:row.id,title:row.title,recent:wasRecent,bytes:row.bytes,qualifiedCatalog:true,
        source:row.source,anatomy:row.anatomy,missingAnatomy:row.missingAnatomy,...(row.sourceLocation?{sourceLocation:row.sourceLocation}:{})});
    };
    for (const filename of [...new Set([...recent, ...this.nativePaths, ...prepared])]) {
      try {
        const catalogRow = catalogAliases.get(path.resolve(filename)) || catalogAliases.get(await fs.realpath(filename));
        if (catalogRow) {addCatalog(catalogRow,recent.includes(filename));continue;}
        const stat = await fs.stat(filename);
        if (!stat.isFile()) continue;
        const id = require('node:crypto').createHash('sha256').update(filename).digest('hex');
        const native=path.basename(filename)==='patient-anatomy.json';
        const doc=native?await readMetadata(filename,16*1024*1024):null;
        if(native&&doc.schema!=='patient-anatomy-v1')continue;
        this.paths.set(id, filename);
        rows.push({id, title: native?doc.patient_id:path.basename(filename, '.ressectionlab'), recent: recent.includes(filename), bytes: stat.size, ...(native?{native:true,timepoint:doc.study_timepoint??doc.series?.[0]?.timepoint,scanCount:doc.series?.length??0}:filename.endsWith('.rsworkspace')?{portable:true}:{})});
      } catch (error) { if (error.code !== 'ENOENT' && !(error instanceof SyntaxError)) throw error; }
    }
    for (const row of catalog) addCatalog(row,recent.some(filename=>row.aliases.includes(filename)));
    return rows;
  }
  async resolve(id) {
    if (typeof id !== 'string' || !/^(?:catalog-)?[a-f0-9]{64}$/.test(id))
      throw new Error('Select a saved case from the current library.');
    await this.list();
    if (typeof id !== 'string' || !this.paths.has(id)) throw new Error('Select a saved case from the current library.');
    const selected=this.paths.get(id);
    return selected?.catalog ? this.catalog.resolve(id) : selected;
  }
}
module.exports = {CaseLibrary};
