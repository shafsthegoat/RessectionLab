'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const {CaseLibrary} = require('./case-library.cjs');

test('case library retains only successful local selections and resolves opaque listed IDs', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(),'rsl-library-test-'));
  t.after(()=>fs.rm(root,{recursive:true,force:true}));
  const prepared=path.join(root,'prepared'),user=path.join(root,'user');
  await fs.mkdir(prepared);
  const a=path.join(prepared,'Public case A.ressectionlab'),b=path.join(root,'Saved session.ressectionlab');
  await fs.writeFile(a,'source');await fs.writeFile(b,'saved');
  await fs.writeFile(path.join(prepared,'catalog.json'),'{}');
  const library=new CaseLibrary(user,prepared);
  assert.equal((await library.list()).length,1);
  await library.remember(b);await library.remember(b);
  const reopened=new CaseLibrary(user,prepared),rows=await reopened.list();
  assert.deepEqual(rows.map(row=>row.title),['Saved session','Public case A']);
  assert.equal(await reopened.resolve(rows[0].id),await fs.realpath(b));
  assert.equal(rows[0].recent,true);
  await assert.rejects(()=>reopened.resolve('../outside'),/current library/);
  await assert.rejects(()=>reopened.remember(path.join(root,'missing')),/ENOENT/);
  assert.equal((await reopened.read()).length,1);
  await fs.unlink(b);
  assert.deepEqual((await reopened.list()).map(row=>row.title),['Public case A']);
});
