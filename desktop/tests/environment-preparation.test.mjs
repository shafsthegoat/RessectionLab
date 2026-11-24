import {after,before,test} from 'node:test';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import React from 'react';
import {createServer} from 'vite';

let server,Panel;
before(async()=>{
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),logLevel:'error',
    server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({EnvironmentPreparationPanel:Panel}=await server.ssrLoadModule('/src/EnvironmentPreparationPanel.tsx'));
});
after(async()=>server?.close());

function harness(){
  const slots=[],cleanups=[];
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  let index=0;
  const dispatcher={
    useState(initial){
      const slot=index++;
      if(!(slot in slots))slots[slot]=typeof initial==='function'?initial():initial;
      return [slots[slot],value=>{slots[slot]=value;}];
    },
    useRef(value){const slot=index++;return slots[slot]??=( {current:value} );},
    useEffect(effect){const slot=index++;if(!(slot in slots)){slots[slot]=true;cleanups.push(effect());}},
  };
  return {
    render(props){
      const previous=internals.H;internals.H=dispatcher;index=0;
      try{return Panel(props);}finally{internals.H=previous;}
    },
    unmount(){for(const cleanup of cleanups)cleanup?.();},
  };
}
function find(tree,match){
  if(!tree||typeof tree!=='object')return;
  if(Array.isArray(tree))return tree.map(child=>find(child,match)).find(Boolean);
  return match(tree)?tree:find(tree.props?.children,match);
}
const control=(tree,name)=>find(tree,node=>node.props?.['aria-label']===name);
const button=(tree,label)=>find(tree,node=>node.type==='button'&&node.props.children===label);
const change=(tree,name,value)=>control(tree,name).props.onChange({target:{value}});

const reviewed={brainLayerId:'brain',vesselLayerId:'tof-score',scope:'reviewed-crop',centerRasMm:[12,34,56]};
const cursor={brainLayerId:'brain',vesselLayerId:'cta-score',scope:'cursor'};
function fixture(pairs){
  const rows=[['brain','brain','ready'],['tumor','synthetic-lesion','ready'],['bone','skull','ready'],
    ['vessels','published-arteries','ready'],['vessels','cta-score','ready'],['vessels','tof-score','estimated_alignment']];
  const calls=[];
  const props={binding:{collectionId:'study',document:{snapshot_id:'exact-revision',layers:rows.map(([,id])=>({id,label:id}))}},
    options:{referenceIssue:null,nativeWall:{referenceIssue:null,pairs},layers:rows.map(([role,layerId,status])=>({role,layerId,status}))},
    patientCursor:null,busy:false,onAnatomy(){},onPrepare:async (args,isCurrent)=>calls.push({args,isCurrent})};
  return {props,calls};
}

test('a unique reviewed pair prepares directly and leaves the separate standard defaults available',async()=>{
  const {props,calls}=fixture([reviewed]),ui=harness();
  let tree=ui.render(props);
  assert.equal(control(tree,'Rehearsal environment').props.value,'native-wall');
  assert.equal(control(tree,'Environment tumor layer'),undefined);
  const prepare=button(tree,'Prepare reviewed local wall');
  assert.equal(prepare.props.disabled,false);
  prepare.props.onClick();
  await new Promise(setImmediate);
  assert.deepEqual(calls[0].args,{collectionId:'study',snapshotId:'exact-revision',layers:{brain:'brain',vessels:'tof-score'},
    allowUnreviewedRegistration:false,nativeWallCenter:[12,34,56]});
  assert.equal(calls[0].isCurrent(),true);
  change(ui.render(props),'Rehearsal environment','standard');
  tree=ui.render(props);
  assert.equal(control(tree,'Environment tumor layer').props.value,'synthetic-lesion');
  button(tree,'Prepare standard environment').props.onClick();
  await new Promise(setImmediate);
  assert.deepEqual(calls[1].args,{collectionId:'study',snapshotId:'exact-revision',layers:{brain:'brain',tumor:'synthetic-lesion',
    bone:'skull',vessels:'published-arteries'},allowUnreviewedRegistration:false});
  ui.unmount();
  assert.equal(calls[0].isCurrent(),false);
});

test('multiple pairs require selection and a cursor pair uses only a finite patient cursor',async()=>{
  const {props,calls}=fixture([reviewed,cursor]),ui=harness();
  let tree=ui.render(props);
  assert.equal(control(tree,'Native wall source pair').props.value,'');
  assert.equal(button(tree,'Prepare native wall at cursor').props.disabled,true);
  change(tree,'Native wall source pair',JSON.stringify(['brain','cta-score']));
  tree=ui.render({...props,patientCursor:[0,NaN,2]});
  assert.equal(button(tree,'Prepare native wall at cursor').props.disabled,true);
  tree=ui.render({...props,patientCursor:[0,1,2]});
  assert.equal(button(tree,'Prepare native wall at cursor').props.disabled,false);
  button(tree,'Prepare native wall at cursor').props.onClick();
  await new Promise(setImmediate);
  assert.deepEqual(calls[0].args.nativeWallCenter,[0,1,2]);
  assert.deepEqual(calls[0].args.layers,{brain:'brain',vessels:'cta-score'});
  ui.render({...props,busy:true});
  assert.equal(calls[0].isCurrent(),false);
});

test('an unavailable local pair and the planned approach cannot use the same prepare action',()=>{
  const {props}=fixture([]),ui=harness();
  let tree=ui.render(props);
  assert.equal(control(tree,'Rehearsal environment').props.value,'standard');
  change(tree,'Rehearsal environment','native-wall');
  tree=ui.render(props);
  assert.equal(button(tree,'Prepare native wall at cursor').props.disabled,true);
  const planned=harness().render({...fixture([reviewed]).props,approach:{targetLayerId:'synthetic-lesion'}});
  assert.equal(control(planned,'Rehearsal environment'),undefined);
  assert.equal(button(planned,'Prepare reviewed local wall'),undefined);
  assert.equal(button(planned,'Inspect approach in engine').props.disabled,false);
});
