'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {secondaryWindowOptions,isSecondaryDisplayConnected}=require('./secondary-display.cjs');
const display=(id,x,y,width,height)=>({id,workArea:{x,y,width,height}});
test('secondary launch selects a nonprimary work area before showing, including negative coordinates',()=>{
  const primary=display(5,0,0,2560,1400),secondary=display(9,-1920,-100,1920,1080);
  const result=secondaryWindowOptions([primary,secondary],5);
  assert.deepEqual(result,{displayId:9,options:{x:-1690,y:-50,width:1460,height:980,minWidth:1050,minHeight:720}});
  assert.equal(isSecondaryDisplayConnected([primary,secondary],5,9),true);
  assert.equal(isSecondaryDisplayConnected([primary],5,9),false);
  assert.equal(isSecondaryDisplayConnected([primary,secondary],9,9),false);
});
test('secondary launch refuses one display and fits small nonprimary displays',()=>{
  assert.throws(()=>secondaryWindowOptions([display(1,0,0,2000,1000)],1),/requires a connected nonprimary display/);
  const result=secondaryWindowOptions([display(1,0,0,2000,1000),display(2,2000,0,900,600)],1);
  assert.deepEqual(result.options,{x:2000,y:0,width:900,height:600,minWidth:900,minHeight:600});
  assert.throws(()=>secondaryWindowOptions([display(2,0,0,0,500)],1),/no usable work area/);
});
