exec(open('testcases/outputs/full-case-rehearsal-20261020/geometry/inspect_primary_candidates.py').read().split('r={')[0])
owner=e.occupancy();labels=s.labels.ravel();target=labels==2;badrecords=[]
for ii,jj in [(119,125),(118,125),(119,124),(121,127),(122,126)]:
 ray=np.flatnonzero(s.labels[ii,jj]==2);bottom=int(ray.min());outside=-366.25
 for radius in [.6,.8,1.,1.3,1.5]:
  initial=s.points([np.ravel_multi_index((ii,jj,bottom),s.shape)])[0];initial[2]+=radius-.3
  def check(end):
   a=end.copy();b=end.copy();b[2]=outside+2
   w=set(owner.material_ids(e.cells(a,b,radius+.17)));b[2]=outside;a[2]=end[2];effect=set(owner.material_ids(e.cells(a,b,radius)));a[2]=end[2]+2;b[2]=outside+40;shafts=set(owner.material_ids(e.cells(a,b,.67)))
   bad=[c for c in w if not target[c]]
   return w,effect,shafts,bad
  w,ef,sh,bad=check(initial);clipped=initial.copy()
  if bad:clipped[2]=max(clipped[2],float(s.points(bad)[:,2].max())+radius+.7)
  w,ef,sh,bad=check(clipped)
  badrecords.append({'xy':[ii,jj],'radius':radius,'endZ':float(clipped[2]),'actualEffectTarget':len(ef),'workingNontarget':len(bad),'workingNontargetHighest':None if not bad else s.points(bad).max(0).tolist(),'shaftNotEffect':len(sh-ef),'shaftNotEffectXYZ':s.points(list(sh-ef)[:3]).tolist() if sh-ef else None})
print(json.dumps(badrecords,indent=2));(out/'candidate_diagnostics.json').write_text(json.dumps(badrecords,indent=2))
