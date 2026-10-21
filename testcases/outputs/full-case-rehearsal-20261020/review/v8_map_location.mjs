// Offline source-map lookup using bundled Node only. Input line/column are
// one-based positions printed in V8 function names; no browser is contacted.
import {readFileSync} from 'node:fs';
import {SourceMap} from 'node:module';
import assert from 'node:assert/strict';
if(process.argv[2]==='--self-test'){
  const result=new SourceMap({version:3,sources:['tiny.ts'],names:[],mappings:'AAAA'}).findEntry(0,0);
  assert.equal(result.originalSource,'tiny.ts');assert.equal(result.originalLine,0);assert.equal(result.originalColumn,0);
  console.log(JSON.stringify({status:'passed',syntheticOnly:true}));
}else{
  const [filename,lineText,columnText]=process.argv.slice(2),line=Number(lineText),column=Number(columnText);
  if(!filename||!Number.isInteger(line)||line<1||!Number.isInteger(column)||column<1)throw Error('Usage: node v8_map_location.mjs bundle.js.map LINE COLUMN (one-based)');
  const result=new SourceMap(JSON.parse(readFileSync(filename,'utf8'))).findEntry(line-1,column-1);
  if(!result.originalSource)throw Error('No mapping for generated location');
  console.log(JSON.stringify({...result,originalLine:result.originalLine+1,originalColumn:result.originalColumn+1,
    generatedLine:result.generatedLine+1,generatedColumn:result.generatedColumn+1,coordinateConvention:'one-based output'},null,2));
}
