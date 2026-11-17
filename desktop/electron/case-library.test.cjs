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

test('native collections retain distinct source identities and exclude invalid or unavailable files', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rsl-native-library-test-'));
  t.after(() => fs.rm(root, {recursive: true, force: true}));
  const paths = [];
  for (const [name, content] of Object.entries({
    first: JSON.stringify({schema: 'patient-anatomy-v1', patient_id: 'same-patient'}),
    second: JSON.stringify({schema: 'patient-anatomy-v1', patient_id: 'same-patient'}),
    malformed: '{', wrong: JSON.stringify({schema: 'unrelated'}),
  })) {
    const directory = path.join(root, name);
    await fs.mkdir(directory);
    const filename = path.join(directory, 'patient-anatomy.json');
    await fs.writeFile(filename, content);
    paths.push(filename);
  }
  const library = new CaseLibrary(path.join(root, 'user'), path.join(root, 'absent'), paths);
  const rows = await library.list();
  assert.equal(rows.length, 2);
  assert.ok(rows.every(row => row.native && row.title === 'same-patient'));
  assert.notEqual(rows[0].id, rows[1].id);
  assert.equal(await library.resolve(rows[0].id), paths[0]);
  assert.equal(await library.resolve(rows[1].id), paths[1]);
  const invalidId = require('node:crypto').createHash('sha256').update(paths[3]).digest('hex');
  await assert.rejects(() => library.resolve(invalidId), /current library/);
  await fs.unlink(paths[0]);
  await assert.rejects(() => library.resolve(rows[0].id), /current library/);
  assert.equal((await library.list()).length, 1);
});

test('prepared HDD links and a canonical recent selection remain one stable local card', async t => {
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'rsl-hdd-links-'));
  t.after(()=>fs.rm(root,{recursive:true,force:true}));
  const prepared=path.join(root,'prepared'),external=path.join(root,'external');
  await fs.mkdir(prepared);await fs.mkdir(external);
  const source=path.join(external,'EDEN2020-CTRL_0866-SYNTHETIC-01-v1.ressectionlab');
  await fs.writeFile(source,'native source');
  await fs.symlink(source,path.join(prepared,path.basename(source)));
  await fs.symlink(source,path.join(prepared,'second-link.ressectionlab'));
  const library=new CaseLibrary(path.join(root,'user'),prepared);
  const before=await library.list();assert.equal(before.length,1);
  await library.remember(source);
  const after=await library.list();assert.equal(after.length,1);
  assert.equal(after[0].id,before[0].id);assert.equal(after[0].recent,true);
  assert.equal(await library.resolve(after[0].id),await fs.realpath(source));
  assert.equal(after[0].reviewedExternal,undefined);
  await fs.unlink(source);
  assert.deepEqual(await library.list(),[]);
});
