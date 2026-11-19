'use strict';
// Electron SCRIPT disposable-native-manifest arm-A.ressectionlab arm-B.ressectionlab output
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {app, dialog, ipcMain} = require('electron');

const [native, ...files] = process.argv.slice(2, 5).map(p => fs.realpathSync(p));
const output = path.resolve(process.argv[5]);
const testRoot = fs.realpathSync(path.resolve(__dirname, '../../testcases/outputs')) + path.sep;
assert(native.startsWith(testRoot) && output.startsWith(testRoot) && files.length === 2);
fs.mkdirSync(output, {recursive: true});
assert(!fs.existsSync(path.join(output, 'comparison.json')), 'Choose a fresh evidence directory');
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'recorded-strategy-'));
app.setPath('userData', profile);
if (process.platform === 'darwin') app.setActivationPolicy('accessory');
process.argv.push('--automation-hidden', '--anatomy', native);
delete process.env.RESSECTIONLAB_DEV_URL;
let compared;
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, handler) => handle(channel, async (...args) => {
  const result = await handler(...args);
  if (channel === 'research:compareBoundSurgicalReplays') compared = result;
  return result;
});
dialog.showOpenDialog = async (...args) => {
  assert.equal(args.at(-1).title, 'Compare two recorded procedures');
  return {canceled:false, filePaths: files};
};
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

app.on('browser-window-created', (_, win) => {
  let shown = false;
  win.on('show', () => { shown = true; });
  win.webContents.once('did-finish-load', async () => {
    const js = code => win.webContents.executeJavaScript(code);
    const click = async label => js(`(()=>{const e=[...document.querySelectorAll('button')]
      .find(n=>n.innerText.trim()===${JSON.stringify(label)}&&!n.disabled);
      if(!e)return false;e.scrollIntoView({block:'center'});e.click();return true})()`);
    const wait = async (probe, label) => {
      const until = Date.now()+420000;
      do {
        if(await probe())return;
        const error = await js("document.querySelector('[aria-label=\"Recorded procedure comparison\"] [role=alert]')?.innerText");
        if(error)throw Error(error);
        await delay(160);
      } while(Date.now()<until);
      throw Error('Timed out waiting for '+label);
    };
    try {
      win.setContentSize(1440,900);
      win.webContents.setBackgroundThrottling(false);
      await wait(()=>click('Plan'),'native plan');
      await wait(()=>js("document.querySelector('.rl-viewer')?.dataset.rendererReady==='true'"),'native view');
      await wait(()=>click('Rehearse'),'rehearsal');
      await wait(()=>click('Compare two recordings'),'recorded comparison');
      await wait(()=>js("!!document.querySelector('.recorded-comparison table')"),'verified table');
      assert.equal(compared.arms.length,2);
      assert.notEqual(compared.arms[0].actionsHash,compared.arms[1].actionsHash);
      const rows = await js("[...document.querySelectorAll('.recorded-comparison tr')].map(r=>[...r.children].map(c=>c.textContent.trim()))");
      for(const [index,arm] of compared.arms.entries()) {
        assert(rows[0][index+1].includes(arm.name));
        for(const [label,amount] of [['Target removed¹',arm.targetRemovedMm3],['Whole target remaining',arm.residualMm3],['Other tissue removed',arm.otherRemovedMm3]]) {
          const cell=rows.find(row=>row[0]===label)[index+1];
          assert.equal(Number(cell.replaceAll(',','').split(' ')[0]), Number(amount.toFixed(2)));
        }
      }
      for(const [width,height] of [[1440,900],[1050,720]]) {
        win.setContentSize(width,height);
        await wait(()=>js('innerWidth==='+width),'window size');
        await js("document.querySelector('.recorded-comparison table').scrollIntoView({block:'center'})");
        await delay(300);
        const capture=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
        fs.writeFileSync(path.join(output,'comparison-'+width+'.png'),capture.toPNG());
      }
      assert(!shown && !win.isVisible());
      fs.writeFileSync(path.join(output,'comparison.json'), JSON.stringify({compared,rows,passed:true},null,2)+'\n');
      app.exit(0);
    } catch(error) {
      console.error(error);
      app.exit(1);
    }
  });
});
app.on('will-quit',()=> fs.rmSync(profile,{recursive:true,force:true}));
require('./main.cjs');
