type XYZ=[number,number,number];

/** Shared geometric route displacement and the derivative of its scalar weight. */
export function retractionField(opening:number,station:number){
  const wx=(opening*.62)**2,wy=(opening*.55)**2,wz=(opening*.85)**2;
  const direction:XYZ=[.40*opening,0,-.31*opening];
  return (x:number,y:number,z:number)=>{
    x-=.16*opening;z-=station;
    const weight=Math.exp(-1.65*(x*x/wx+y*y/wy+z*z/wz));
    return {direction,shift:[direction[0]*weight,0,direction[2]*weight] as XYZ,
      gradient:[-3.3*weight*x/wx,-3.3*weight*y/wy,-3.3*weight*z/wz] as XYZ};
  };
}
