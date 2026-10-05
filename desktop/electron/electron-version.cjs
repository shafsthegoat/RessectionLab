'use strict';

const VERSION = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$/;

/**
 * Bind the installed runtime to the captured lock's resolved root dependency.
 * This deliberately accepts only pnpm's current canonical v9 single-importer
 * format. Unsupported YAML syntax fails closed; this is not a YAML parser.
 */
function assertLockedElectronVersion(installedVersion, lockfileText) {
  const invalid = reason => { throw new Error(`Cannot bind Electron to captured pnpm lock: ${reason}`); };
  if (typeof installedVersion !== 'string' || !VERSION.test(installedVersion)) invalid('invalid installed version');
  if (typeof lockfileText !== 'string' || lockfileText.includes('\t')) invalid('unsupported lock text');
  const lines = lockfileText.replace(/\r\n/g, '\n').split('\n').map(line => line.trimEnd());
  const headers = [];
  const seenHeaders = new Set();
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (!line || line.startsWith(' ')) continue;
    const header = /^(lockfileVersion|settings|importers|packages|snapshots):/.exec(line)?.[1];
    if (!header || seenHeaders.has(header)) invalid('unsupported or duplicate top-level field');
    if (header === 'lockfileVersion') {
      if (!/^lockfileVersion: (?:'9\.0'|"9\.0"|9\.0)$/.test(line)) invalid('pnpm v9 is required');
    } else if (line !== `${header}:`) invalid('unsupported top-level mapping');
    seenHeaders.add(header);
    headers.push({ name: header, index });
  }
  if (!seenHeaders.has('lockfileVersion') || !seenHeaders.has('importers')) invalid('missing version or importers');
  const position = headers.findIndex(header => header.name === 'importers');
  const block = lines.slice(headers[position].index + 1, headers[position + 1]?.index ?? lines.length).filter(Boolean);
  if (block[0] !== '  .:' || block.filter(line => /^ {2}\S/.test(line)).length !== 1) invalid('exactly one root importer is required');
  const sections = new Set();
  const records = [];
  let section;
  let record;
  for (const line of block.slice(1)) {
    const sectionMatch = /^ {4}(dependencies|devDependencies|optionalDependencies):$/.exec(line);
    if (sectionMatch) {
      section = sectionMatch[1];
      if (sections.has(section)) invalid('duplicate dependency section');
      sections.add(section);
      record = undefined;
      continue;
    }
    const dependencyMatch = /^ {6}(?:([@A-Za-z0-9._/-]+)|'([@A-Za-z0-9._/-]+)'|"([@A-Za-z0-9._/-]+)"):$/.exec(line);
    if (dependencyMatch && section) {
      const name = dependencyMatch[1] || dependencyMatch[2] || dependencyMatch[3];
      if (records.some(item => item.section === section && item.name === name)) invalid('duplicate dependency entry');
      record = { name, section, fields: new Map() };
      records.push(record);
      continue;
    }
    const fieldMatch = /^ {8}(specifier|version): (.+)$/.exec(line);
    if (!fieldMatch || !record) invalid('unsupported importer structure');
    if (record.fields.has(fieldMatch[1])) invalid('duplicate dependency resolution field');
    record.fields.set(fieldMatch[1], fieldMatch[2]);
  }
  const candidates = records.filter(item => item.name === 'electron');
  if (candidates.length !== 1 || candidates[0].section !== 'devDependencies') invalid('exactly one root Electron devDependency is required');
  const fields = candidates[0].fields;
  if (!fields.has('specifier') || !fields.has('version')) invalid('Electron resolved version is missing');
  const resolved = fields.get('version');
  if (!VERSION.test(resolved)) invalid('Electron must resolve to an exact registry version');
  if (resolved !== installedVersion) invalid(`installed ${installedVersion} differs from locked ${resolved}`);
  return installedVersion;
}

module.exports = { assertLockedElectronVersion };
