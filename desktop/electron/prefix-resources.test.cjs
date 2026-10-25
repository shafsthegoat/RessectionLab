'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {PrefixResources,treeRss,MAX_RSS}=require('./prefix-resources.cjs');
test('resource sample includes both complete trees, excludes unrelated processes',()=>{
 assert.equal(treeRss('10 1 100\n11 10 20\n20 1 200\n21 20 30\n22 21 5\n99 1 99999',[10,20]),355*1024);
 assert.throws(()=>treeRss('10 1 100',[10,20]),/no longer running/);
 assert.throws(()=>treeRss('10 1 bad',[10,20]),/Invalid process/);
});
test('aggregate cap catches individually small workers and fails once',async()=>{
 const failures=[];const rss=Math.floor(MAX_RSS/1024/2)+1;
 const budget=new PrefixResources({pids:[10,20],onFailure:e=>failures.push(e),sample:async()=>`10 1 ${rss}\n20 1 ${rss}`});
 await assert.rejects(budget.start(),/aggregate 3 GiB/);await budget.close();
 assert.equal(failures.length,1);assert.equal(budget.statistics.samples,1);assert.ok(budget.statistics.peakAggregatePythonTreeRssBytes>MAX_RSS);
});
test('wall deadline applies while sample is awaited and close drains a pending sample',async()=>{
 let clock=0,release;const failures=[];
 const budget=new PrefixResources({pids:[10,20],maxWallMs:100,now:()=>clock,onFailure:e=>failures.push(e),sample:()=>new Promise(r=>{release=r;})});
 const checking=budget.check();clock=101;release('10 1 1\n20 1 1');await assert.rejects(checking,/wall-time/);assert.equal(failures.length,1);
 await budget.close();
 let complete;const closing=new PrefixResources({pids:[10,20],onFailure:()=>assert.fail('closed monitor reported failure'),sample:()=>new Promise(r=>{complete=r;})});
 const pending=closing.check();const closed=closing.close();complete('10 1 1\n20 1 1');await Promise.all([pending,closed]);assert.equal(closing.statistics.samples,0);
});
