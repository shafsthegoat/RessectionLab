import * as THREE from 'three';
import type {SurgicalTool, SurgicalSlot} from '../surgical-types';

type Shade = THREE.MeshStandardMaterial;
const yAxis=new THREE.Vector3(0,1,0);

/** Native local capsules describe the moving limbs, conservative guard and
 * distal shaft. Model the visible metal/insulation inside each capsule. */
export function operativeInstrument(tool:SurgicalTool,slot:SurgicalSlot,ghost:boolean):THREE.Group {
  const group=new THREE.Group();
  group.name='instrument:'+tool.id;
  group.userData.physicsContactGeometry=true;
  const material=(color:string,metalness:number,roughness:number)=>new THREE.MeshStandardMaterial({
    color:ghost?'#8ce8d2':color,metalness,roughness,transparent:ghost,opacity:ghost?.38:1,depthWrite:!ghost,
  });
  const steel=material('#d5dce0',.68,.25), bevel=material('#edf3f4',.62,.18),
    body=material('#75818b',.65,.35), insulation=material('#345c85',.14,.4),
    ceramic=material('#ddc9a6',.25,.39), recessed=material('#263039',.33,.55);
  group.userData.profileIndicator=insulation;
  const metalwork=(parent:THREE.Group,geometry:THREE.BufferGeometry,shade:Shade,y=0)=>{
    const mesh=new THREE.Mesh(geometry,shade);mesh.position.y=y;mesh.castShadow=!ghost;mesh.receiveShadow=!ghost;
    parent.add(mesh);return mesh;
  };
  const solid=(parent:THREE.Group,r:number,length:number,shade:Shade)=>{
    metalwork(parent,new THREE.CapsuleGeometry(r,length,4,16),shade);
  };
  const cylinder=(parent:THREE.Group,r:number,length:number,shade:Shade,y=0)=>
    metalwork(parent,new THREE.CylinderGeometry(r,r,length,24),shade,y);

  for(const [index,segment] of (slot.contactGeometry??[]).entries()){
    const form=tool.physicalProfile!.parts[index];
    const a=new THREE.Vector3(...segment.a), b=new THREE.Vector3(...segment.b);
    const length=a.distanceTo(b),r=segment.radiusMm;
    const frame=new THREE.Group();
    frame.name='instrument:'+form;
    frame.position.copy(a).add(b).multiplyScalar(.5);
    const direction=length?b.clone().sub(a).normalize():new THREE.Vector3(0,0,1);
    frame.quaternion.setFromUnitVectors(yAxis,direction);group.add(frame);

    if(form==='finger-loop'){
      const loop=new THREE.Shape();
      loop.absellipse(0,0,r*.89,length/2+r*.85,0,2*Math.PI,false,0);
      const hole=new THREE.Path();hole.absellipse(0,0,r*.62,length/2+r*.51,0,2*Math.PI,true,0);
      loop.holes.push(hole);
      const mesh=metalwork(frame,new THREE.ExtrudeGeometry(loop,{depth:r*.24,bevelEnabled:false,curveSegments:24}),steel);
      mesh.position.z=-r*.12;
    }else if(form==='motor-handpiece'||form==='ultrasonic-handpiece'||form==='driver-grip'){
      solid(frame,r*.88,length,form==='driver-grip'?ceramic:body);
      for(let j=1;j<8;j++)cylinder(frame,r*.95,length*.012,insulation,length*(j/9-.47));
    }else if(form==='flat-handle'||form==='spring-grip'||form==='lever-arm'){
      const mesh=metalwork(frame,new THREE.CapsuleGeometry(r*.94,length,4,16),steel);
      mesh.scale.z=.42;
      if(form!=='lever-arm')for(let j=0;j<5;j++){
        const ridge=cylinder(frame,r*.96,length*.009,body,length*(j/12-.3));ridge.scale.z=.48;
      }
    }else if(form==='valve-grip'||form==='pencil-grip'){
      solid(frame,r*.88,length,form==='valve-grip'?steel:body);
      for(const fraction of [-.35,-.27,.33])cylinder(frame,r*.96,length*.025,insulation,length*fraction);
    }else if(form==='proximal-shank'||form==='angled-neck'){
      solid(frame,r,length,steel);
    }else if(form==='shaft'){
      const shank=cylinder(frame,r*.95,length,steel);
      if(tool.id==='scalpel'){
        shank.scale.x=.58;
        shank.scale.z=.34;
      }
      // Short bands at the rear of the declared distal shaft. They never
      // extend the modeled working length into an unreviewed handle.
      for(const y of [.9,1.5,2.1])cylinder(frame,r,.15,body,-length/2+y).scale.copy(shank.scale);
    }else if(form==='fluted-burr'){
      const rotor=new THREE.Group();frame.add(rotor);
      rotor.name='instrument:profile-burr';
      metalwork(rotor,new THREE.SphereGeometry(r*.95,24,18),steel);
      // Six curved flutes share a bounded spherical head, each inside its cover.
      for(let j=0;j<6;j++){
        const curve=[];
        for(let k=1;k<16;k++){
          const theta=Math.PI*k/16,phi=j*Math.PI/3+.85*Math.cos(theta);
          curve.push(new THREE.Vector3(r*.95*Math.sin(theta)*Math.cos(phi),
            r*.95*Math.cos(theta),r*.95*Math.sin(theta)*Math.sin(phi)));
        }
        metalwork(rotor,new THREE.TubeGeometry(new THREE.CatmullRomCurve3(curve),18,r*.035,5,false),body);
      }
    }else if(form.endsWith('lumen')||form==='irrigation-nozzle'){
      // A true visible annular end and inner wall. The ball-shaped CUSA
      // acoustic/contact envelope remains authoritative for tissue work.
      const len=length||r*.95,outer=r*(length?.94:.74),inner=outer*.68;
      const shell=new THREE.LatheGeometry([new THREE.Vector2(inner,-len/2),new THREE.Vector2(outer,-len/2),
        new THREE.Vector2(outer,len/2),new THREE.Vector2(inner,len/2),new THREE.Vector2(inner,-len/2)],32);
      metalwork(frame,shell,form==='aspiration-lumen'?ceramic:steel);
      cylinder(frame,inner*.995,.05,recessed,-len*.35);
    }else if(['scissor-edge','blade-heel','blade-point','side-cutter','driver-bit'].includes(form)){
      if(form==='side-cutter'||form==='driver-bit'){
        const cutter=metalwork(frame,new THREE.CylinderGeometry(r*.87,r,length,form==='driver-bit'?6:8),body);
        cutter.rotation.y=Math.PI/8;
        if(form==='side-cutter')for(let j=0;j<3;j++)cylinder(frame,r*.94,.11,bevel,length*(j/4-.3));
      }else{
        // Tapered and beveled steel edge, constructed in the authored X/Z
        // plane so a change in the shaft's roll also turns the cutting plane.
        const tangent=direction.clone(),normal=new THREE.Vector3(0,1,0);
        const edge=new THREE.Vector3().crossVectors(normal,tangent).normalize();
        const vertices:number[]=[];
        for(const [t,w] of [[0,.90],[1,form==='blade-point'?.04:.54]]){
          const center=a.clone().lerp(b,t);
          for(const [side,h] of [[-1,0],[.35,.19],[1,0],[.35,-.19]])
            vertices.push(...center.clone().addScaledVector(edge,side*w*r).addScaledVector(normal,h*r).toArray());
        }
        const blade=new THREE.BufferGeometry();blade.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));
        blade.setIndex([0,4,1,1,4,5,1,5,2,2,5,6,2,6,3,3,6,7,3,7,0,0,7,4,0,1,3,1,2,3,4,7,5,5,7,6]);
        blade.computeVertexNormals();frame.position.set(0,0,0);frame.quaternion.identity();
        metalwork(frame,blade,bevel);
      }
    }else if(form==='paddle-strip'){
      // The strips meet edge-to-edge and fit inside each radius-.54 cover.
      metalwork(frame,new THREE.BoxGeometry(1,length,.36),steel);
    }else if(form==='guard-post'||form==='guard-foot'){
      solid(frame,r,length,ceramic);
    }else if(form==='insulated-arm'||form==='forceps-arm'){
      solid(frame,r*.95,length,form==='insulated-arm'?insulation:steel);
    }else if(['serrated-jaw','dressing-jaw','gripping-cup','clip-seat','needle-jaw','fine-jaw'].includes(form)){
      solid(frame,r*.68,length,steel);
      const grooved=['serrated-jaw','gripping-cup','clip-seat','dressing-jaw'].includes(form);
      if(grooved)for(let j=0;j<4;j++)cylinder(frame,r*.94,length*.065,form==='clip-seat'?insulation:body,length*(.17+j*.085));
      if(form==='needle-jaw')cylinder(frame,r*.86,length*.44,ceramic,length*.21);
    }else{
      solid(frame,r,length,form==='electrode'?ceramic:form==='chuck'||form==='pivot'?body:steel);
    }
  }
  return group;
}
