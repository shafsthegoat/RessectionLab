'use strict';
// Electron SCRIPT disposable-native-manifest arm-A.ressectionlab arm-B.ressectionlab output
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {createHash}=require('node:crypto');
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
const rendererErrors=[];
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
  win.webContents.on('console-message',(_,message)=>{if(message.level==='error')rendererErrors.push(message.message);});
  win.webContents.on('did-fail-load',(_,code,description)=>rendererErrors.push(code+' '+description));
  win.webContents.on('render-process-gone',(_,details)=>rendererErrors.push(details.reason));
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
      const sections=compared.sections.planes, captures=[];
      assert.equal(sections.length,2);
      assert.notEqual(compared.arms[0].endpointStateId,compared.arms[1].endpointStateId);
      assert.equal(compared.arms[0].baselineStateId,compared.arms[1].baselineStateId);
      const section=`document.querySelector('[aria-label="Matched endpoint sections"]')`;
      for(const [width,height] of [[1440,900],[1050,720]]) {
        win.setContentSize(width,height);
        await wait(()=>js('innerWidth==='+width+'&&innerHeight==='+height),'window size');
        const capture=async name=>{
          await delay(250);
          const image=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
          fs.writeFileSync(path.join(output,name+'.png'),image.resize({width,height}).toPNG());
        };
        for(const [index,plane] of sections.entries()){
          await wait(()=>click(index?'Along B':'Along A'),'shared section plane');
          await js("document.querySelector('.recorded-comparison').scrollTop=0");
          await wait(()=>js(`[...(${section}).querySelectorAll('img')].every(img=>img.complete&&img.naturalWidth===${plane.size})`),'decoded endpoint images');
          const panels=await js(`[...(${section}).querySelectorAll('figure')].map(fig=>{const img=fig.querySelector('img'),r=img.getBoundingClientRect();return {state:fig.dataset.endpointState,src:img.src,rect:[r.x,r.y,r.width,r.height],fraction:fig.querySelector('.endpoint-fraction')?.textContent};})`);
          assert.equal(panels.length,2);
          for(const [j,panel] of panels.entries()){
            assert.equal(panel.state,compared.arms[j].endpointStateId);
            assert.equal(panel.src,'data:image/png;base64,'+plane.arms[j].png);
            const [x,y,w,h]=panel.rect;
            assert(x>=0&&y>=0&&x+w<=width+1&&y+h<=height+1&&w>=160&&Math.abs(w-h)<1,'Endpoint panel clipped');
            delete panel.src;
          }
          const name='comparison-'+width+'-along-'+(index?'b':'a');
          await capture(name);captures.push({image:name+'.png',width,height,index,panels});
        }
        await js("document.querySelector('.recorded-comparison table').scrollIntoView({block:'center'})");
        await capture('comparison-'+width);
      }
      assert(!shown&&!win.isVisible());assert.deepEqual(rendererErrors,[]);
      const sha256=f=>createHash('sha256').update(fs.readFileSync(f)).digest('hex');
      const assets=path.resolve(__dirname,'../dist/assets');
      const rendererAssets=Object.fromEntries(fs.readdirSync(assets).filter(name=>/^index-.*\.(js|css)$/.test(name)).map(name=>[name,sha256(path.join(assets,name))]));
      fs.writeFileSync(path.join(output,'comparison.json'),JSON.stringify({compared,rows,captures,rendererErrors,rendererAssets,pid:process.pid,studySha256:sha256(native),
        recordings:files.map(f=>({name:path.basename(f),sha256:sha256(f)})),passed:true},null,2)+'\n');
      app.exit(0);
    } catch(error) {
      console.error(error);
      app.exit(1);
    }
  });
});
app.on('will-quit',()=> fs.rmSync(profile,{recursive:true,force:true}));
require('./main.cjs');
