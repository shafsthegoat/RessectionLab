import * as THREE from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';

// Original, unitless display geometry. +X right, +Y anterior, +Z superior.
// The neck pivot is the origin so head and body can have separate RAS frames.
type Ring = [z:number, width:number, depth:number, x?:number, y?:number];
function loft(rings:Ring[], steps=48, sides=28):THREE.BufferGeometry {
  const positions:number[]=[],indices:number[]=[];
  const interpolate=(i:number,t:number,k:number)=>{
    const p0=rings[Math.max(0,i-1)][k]??0,p1=rings[i][k]??0,
      p2=rings[i+1][k]??0,p3=rings[Math.min(rings.length-1,i+2)][k]??0;
    return .5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t*t+(-p0+3*p1-3*p2+p3)*t*t*t);
  };
  for(let row=0;row<=steps;row++){
    const u=row/steps*(rings.length-1),i=Math.min(rings.length-2,Math.floor(u)),t=u-i;
    const z=interpolate(i,t,0),rx=Math.max(.0001,interpolate(i,t,1)),ry=Math.max(.0001,interpolate(i,t,2)),
      x=interpolate(i,t,3),y=interpolate(i,t,4);
    for(let side=0;side<sides;side++){
      const angle=2*Math.PI*side/sides;
      positions.push(x+rx*Math.cos(angle),y+ry*Math.sin(angle),z);
      if(row<steps){const a=row*sides+side,b=row*sides+(side+1)%sides,c=a+sides,d=b+sides;indices.push(a,b,c,b,d,c);}
    }
  }
  // Close both ends without duplicating the circumferential seam.
  for(const row of [0,steps]){
    const ring=rings[row===0?0:rings.length-1],centre=positions.length/3;
    positions.push(ring[3]??0,ring[4]??0,ring[0]);
    for(let side=0;side<sides;side++){
      const a=row*sides+side,b=row*sides+(side+1)%sides;
      indices.push(...(row===0?[centre,b,a]:[centre,a,b]));
    }
  }
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setIndex(indices);geometry.computeVertexNormals();
  return geometry;
}
function ellipsoid(x:number,y:number,z:number,rx:number,ry:number,rz:number):THREE.BufferGeometry {
  const geometry=new THREE.SphereGeometry(1,20,12);geometry.deleteAttribute('uv');
  geometry.scale(rx,ry,rz);geometry.translate(x,y,z);return geometry;
}
function joined(parts:THREE.BufferGeometry[]):THREE.BufferGeometry {
  const geometry=mergeGeometries(parts);parts.forEach(part=>part.dispose());
  if(!geometry)throw Error('Could not build body orientation guide');
  geometry.computeBoundingSphere();return geometry;
}

/** Two resident meshes, smooth lofts and a readable face/foot profile, no assets. */
export function orientationMannequin(){
  const bodyParts=[loft([
    [-.79,.035,.055],[-.75,.115,.087],[-.65,.155,.105],[-.54,.133,.083],
    [-.40,.146,.091],[-.25,.176,.108],[-.14,.196,.094],[-.085,.163,.074],[-.045,.047,.042],
  ])];
  for(const side of [-1,1]){
    bodyParts.push(
      // Slight arm abduction preserves the torso silhouette in a tiny view.
      ellipsoid(side*.188,0,-.146,.058,.066,.079),
      loft([[-.755,.020,.025,side*.300,.013],[-.68,.031,.033,side*.296,.006],
        [-.57,.043,.043,side*.278,0],[-.43,.040,.040,side*.260,-.006],
        [-.30,.049,.050,side*.233,0],[-.17,.054,.056,side*.197,0]],32,24),
      ellipsoid(side*.304,.013,-.788,.035,.023,.072),
      ellipsoid(side*.276,.034,-.766,.014,.018,.035),
      loft([[-1.47,.025,.031,side*.104,0],[-1.37,.032,.036,side*.104,-.006],
        [-1.22,.052,.054,side*.103,-.014],[-1.09,.046,.047,side*.101,.002],
        [-.96,.059,.064,side*.096,0],[-.82,.077,.080,side*.090,0],[-.69,.075,.081,side*.081,0]],40,28),
      ellipsoid(side*.104,.045,-1.484,.039,.096,.037),
    );
  }
  const headParts=[loft([
    [-.025,.037,.035],[.045,.039,.037],[.071,.041,.045,0,.009],
    [.094,.059,.062,0,.009],[.145,.075,.079],[.201,.077,.082,0,-.005],
    [.248,.055,.061,0,-.006],[.268,.001,.001,0,-.006],
  ],40,32),
  // A quiet nose, chin and ears distinguish front/back without facial texture.
  ellipsoid(0,.082,.142,.016,.025,.025),ellipsoid(0,.060,.092,.030,.015,.015),
  ellipsoid(-.076,-.003,.155,.012,.019,.030),ellipsoid(.076,-.003,.155,.012,.019,.030)];
  const body=new THREE.Mesh(joined(bodyParts),new THREE.MeshStandardMaterial({color:'#aebbc1',roughness:.77,metalness:0}));
  const head=new THREE.Mesh(joined(headParts),new THREE.MeshStandardMaterial({color:'#87d8c0',roughness:.7,metalness:0}));
  body.name='orientation:body';head.name='orientation:head';
  const model=new THREE.Group();model.add(body,head);
  return {model,body,head};
}
