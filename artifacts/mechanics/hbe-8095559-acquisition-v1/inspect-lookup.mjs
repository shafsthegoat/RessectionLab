import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const root = process.cwd();
const source = path.join(root, 'data/mechanics/zenodo-8095559/sample_lookup.xlsx');
const book = await SpreadsheetFile.importXlsx(await FileBlob.load(source));
const overview = await book.inspect({kind: 'workbook,sheet,table', maxChars: 5000, tableMaxRows: 4, tableMaxCols: 10});
console.log(overview.ndjson);
const result = [];
for (let i = 0; i < book.worksheets.items.length; i++) {
  const sheet = book.worksheets.getItemAt(i);
  const range = sheet.getUsedRange();
  result.push({sheet: sheet.name, values: range.values});
}
await fs.writeFile(path.join(root, 'data/mechanics/zenodo-8095559/sample-lookup-metadata.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result.map(item => ({sheet: item.sheet, rows: item.values.length, firstRows: item.values.slice(0, 5)}))));
