import * as THREE from 'three';
import {PLANNING_ROUTE_COLORS,planningRoutePoints,declaredStructureColor} from '../interactive-planning-data';
import type {InteractivePlanningOverlay} from '../interactive-planning-data';
/** Source RAS guides: draft proposals or explicitly bound recorded native paths. */
export function interactivePlanningGroup(overlay:InteractivePlanningOverlay):THREE.Group {
  const group=new THREE.Group();group.name='interactive-approach-planning';
  for(const [index,route] of overlay.routes.entries()){
    const selected=route.selected??route.id===overlay.selectedId,color=PLANNING_ROUTE_COLORS[(route.colorIndex??index)%PLANNING_ROUTE_COLORS.length];
    const geometry=new THREE.BufferGeometry().setFromPoints(planningRoutePoints(route).map(p=>new THREE.Vector3(...p)));
    const line=new THREE.Line(geometry,new THREE.LineBasicMaterial({color,transparent:true,opacity:selected?1:.55,depthTest:false,depthWrite:false}));
    line.name=`approach:${route.id}`;line.renderOrder=20;group.add(line);
    const endpoint=new THREE.Mesh(new THREE.SphereGeometry(selected?1.1:.65,12,8),new THREE.MeshBasicMaterial({color,depthTest:false,depthWrite:false,transparent:true,opacity:selected?1:.7}));
    endpoint.position.set(...route.entryPose.position);endpoint.renderOrder=21;group.add(endpoint);
  }
  if(!overlay.recorded)for(const [name,region,color] of [['target',overlay.constraints.target,'#ff91ae'],['entry',overlay.constraints.entry,'#70e0ff']] as const){
    const geometry=new THREE.SphereGeometry(region.radiusMm,24,16),material=new THREE.MeshBasicMaterial({color,wireframe:true,transparent:true,opacity:.4,depthWrite:false});
    const sphere=new THREE.Mesh(geometry,material);sphere.name=`planning-${name}-region`;sphere.position.set(...region.centerMm);group.add(sphere);
  }
  for(const structure of overlay.limitingStructures){
    const mark=new THREE.Mesh(new THREE.OctahedronGeometry(.9),new THREE.MeshBasicMaterial({color:'#ffb65c',depthTest:false,depthWrite:false}));
    mark.position.set(...structure.positionMm);mark.name=`planning-limit:${structure.name}`;mark.renderOrder=22;group.add(mark);
  }
  if(overlay.recorded){
    const dots=(name:string,points:number[][],color:string,size:number,order:number)=>{
      if(!points.length)return;
      const geometry=new THREE.BufferGeometry().setFromPoints(points.map(p=>new THREE.Vector3(...p as [number,number,number])));
      const material=new THREE.PointsMaterial({color,size,sizeAttenuation:false,depthTest:false,depthWrite:false,transparent:true,opacity:.85});
      const marks=new THREE.Points(geometry,material);marks.name=name;marks.renderOrder=order;group.add(marks);
    };
    dots('recorded-original-target',overlay.recorded.targetPointsRasMm,'#ffb6cd',3,18);
    dots('recorded-residual-target',overlay.recorded.residualPointsRasMm,'#ff6b9e',5,19);
    for(const s of overlay.recorded.structures){
      const rank=s.id==='motor-cortex'?1:s.id==='corticospinal-tract'?2:0;
      dots(`declared-source:${s.id}`,s.pointsRasMm,declaredStructureColor(s.id),s.id==='motor-cortex'?11:s.id==='corticospinal-tract'?6:s.id==='supratentorial-neural'?3:8,23+rank);
    }
  }
  return group;
}
