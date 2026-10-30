#!/usr/bin/env python3
"""Independent conditional closed-loop export checker. No app imports/solves."""
from pathlib import Path
from fractions import Fraction
from itertools import product
import argparse,copy,gzip,hashlib,io,json,math,runpy,sys,time,zipfile
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
KEYS=('cells','shape','nodes','elementNodes','affine','freeDofs','muPa','nu')
CORNERS=np.array(list(product((0.,1.),repeat=3)));EPS=1e-9

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def record_hash(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def finite(v,shape,name):
    a=np.asarray(v);need(a.shape==shape and a.dtype.kind in 'fiu' and np.isfinite(a).all(),'Invalid '+name);return np.asarray(a,float)
def scalar(v,name):return float(finite(v,(),name))
def integer(v,name):need(type(v) is int and v>=0,'Invalid integer '+name);return v
def near(a,b,tol,name):
    a,b=np.asarray(a,float),np.asarray(b,float);need(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid comparison '+name)
    e=float(np.max(np.abs(a-b),initial=0.));need(e<=tol,name+': '+str(e));return e
def bits(a,b,name):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64);need(a.shape==b.shape and np.array_equal(a.view(np.uint64),b.view(np.uint64)),name)
def field_hash(u,z):
    h=hashlib.sha256(b'resectionlab.full-source-fields.v1')
    for a in (np.asarray(u,np.float64).ravel(),np.asarray(z,np.float64).reshape(len(z),-1)):
        h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
def scope(v):
    need(v['physicalValidationPassed'] is False and v['largerDomainPublicActionsAdmitted'] is False,'Scope promotion')
    need(v['physiologicalRL']=='hold' and v['largeScaleRL']=='hold','RL promotion')
    need(type(v['trainingUpdates']) is int and type(v['newAcquiredDatasetBytes']) is int and v['trainingUpdates']==0 and v['newAcquiredDatasetBytes']==0,'Data/training promotion')
def stripped(v):
    if isinstance(v,dict):return {k:stripped(x) for k,x in v.items() if k not in ('wallSeconds','queryWallSeconds','builds','preparationSeconds')}
    if isinstance(v,list):return [stripped(x) for x in v]
    return v

def exact_capacity(before,removed,order,dt,parameters,volume,rate):
    s=copy.deepcopy(before);need(s['schema']=='resectionlab.cusa-contact.v3','Capacity schema');gone=set(removed);need(len(order)==len(set(order)) and not gone.intersection(order),'Capacity owner set')
    need(set(s['progressExact'])==set(s['progressMm3'])==set(s['contactSeconds']),'Capacity ledger keys')
    unit=Fraction.from_float(float(volume))
    for k,v in s['progressExact'].items():
        need(isinstance(v,list) and len(v)==2 and all(type(x) is int for x in v) and v[1]>0,'Capacity rational')
        f=Fraction(*v);need(0<=f<unit and float(f)==s['progressMm3'][k],'Capacity rational/display mismatch')
    stale=[k for k in s['progressExact'] if int(k) in gone];discard=math.fsum(s['progressMm3'][k] for k in stale)
    for k in stale:
        for name in ('progressExact','progressMm3','contactSeconds'):s[name].pop(k)
    s['discardedCapacityMm3']+=discard
    power=scalar(parameters['power'],'power');duty=scalar(parameters['duty'],'duty');need(0<=power<=1 and 0<=duty<=1 and dt>0,'Activation')
    budget=rate*power*duty*dt;remaining=Fraction.from_float(budget);new=[];spent=[]
    for cell in order:
        k=str(cell);old=Fraction(*s['progressExact'][k]) if k in s['progressExact'] else Fraction(0);take=min(unit-old,remaining)
        if take<=0:continue
        remaining-=take;spent.append(float(take));s['contactSeconds'][k]=s['contactSeconds'].get(k,0.)+dt
        after=old+take
        if after==unit:
            new.append(cell)
            for name in ('progressExact','progressMm3','contactSeconds'):s[name].pop(k,None)
        else:s['progressExact'][k]=[after.numerator,after.denominator];s['progressMm3'][k]=float(after)
    s['inputCapacityMm3']+=budget;s['aspiratedSolidMm3']+=len(new)*volume;s['unusedCapacityMm3']+=float(remaining);s['effectiveActivationSeconds']+=power*duty*dt
    events=[{'kind':'cusa_contact','inputCapacityMm3':budget,'allocatedCapacityMm3':math.fsum(spent),'unusedCapacityMm3':float(remaining),'aspiratedSolidMm3':len(new)*volume,'discardedCapacityMm3':discard,'removedCells':new,'pendingCells':len(s['progressExact'])}] if budget or discard else []
    return s,sorted(gone|set(new)),events

class Raster:
    def __init__(self,labels,affine,g,u,removed,soft=None):
        self.labels=labels.ravel();self.shape=labels.shape;self.affine=affine;self.inverse=np.linalg.inv(affine);self.spacing=np.linalg.norm(affine[:3,:3],axis=0)
        need(np.allclose(affine[:3,:3].T@affine[:3,:3],np.diag(self.spacing**2),rtol=0,atol=1e-12),'Nonorthogonal legacy grid unsupported')
        self.removed=set(removed);self.lookup={int(c):i for i,c in enumerate(g['cells'])};centers=u[g['elementNodes']].mean(axis=1)
        ijk=np.array(np.unravel_index(g['cells'],self.shape)).T;voxel=np.rint(ijk+centers@self.inverse[:3,:3].T).astype(int)
        need(np.all((voxel>=0)&(voxel<np.array(self.shape))),'Raster FOV');dest=np.ravel_multi_index(voxel.T,self.shape)
        self.cleared=self.removed|set(self.lookup);self.members={}
        for owner,target in zip(g['cells'],dest):
            owner,target=int(owner),int(target)
            if self.labels[target] and target not in self.cleared:
                need(soft is None or int(self.labels[target]) in soft,'Raster blocking material collision');self.members.setdefault(target,set()).add(target)
            self.members.setdefault(target,set()).add(owner)
        self.centers={int(c):point for c,point in zip(g['cells'],ijk@affine[:3,:3].T+affine[:3,3]+centers)}
    def grid(self,point,radius):
        p=finite(point,(3,),'point');v=p@self.inverse[:3,:3].T+self.inverse[:3,3];extent=(radius+EPS)/self.spacing+.5+EPS
        lo=np.maximum(0,np.ceil(v-extent).astype(int));hi=np.minimum(np.array(self.shape)-1,np.floor(v+extent).astype(int))
        if np.any(lo>hi):return np.empty(0,np.int64)
        ids=np.stack(np.meshgrid(*[np.arange(a,b+1) for a,b in zip(lo,hi)],indexing='ij'),axis=-1).reshape(-1,3)
        excess=np.maximum(np.abs((v-ids)*self.spacing)-self.spacing/2,0.)
        return np.ravel_multi_index(ids[np.sqrt(np.sum(excess**2,axis=1))<=radius+EPS].T,self.shape)
    def contacts(self,point):
        owners=set()
        for v in self.grid(point,1.5):
            v=int(v)
            if v in self.members:owners.update(self.members[v])
            elif v not in self.cleared and self.labels[v]:owners.add(v)
        return sorted(owners-self.removed)
    def ordered(self,point,soft):
        p=np.asarray(point,float);ids=[i for i in self.contacts(p) if int(self.labels[i]) in soft]
        def distance(i):
            center=self.centers.get(i)
            if center is None:center=np.array(np.unravel_index(i,self.shape))@self.affine[:3,:3].T+self.affine[:3,3]
            return float(np.sum((center-p)**2)),i
        return sorted(ids,key=distance)
    def limiter(self,start,requested,soft):
        a,b=np.array(start,float),np.array(requested,float);queries=[]
        def query(point):
            order=self.ordered(point,soft);queries.append({'positionMm':point.tolist(),'candidateSourceIds':self.contacts(point),'orderedSourceIds':order});return order
        initial=query(a)
        if initial:return a,initial,queries
        n=max(1,math.ceil(float(np.linalg.norm(b-a))/(float(self.spacing.min())/4)));need(n<10000,'Limiter work bound');low=a.copy()
        for i in range(1,n+1):
            high=a+(b-a)*(i/n);ids=query(high)
            if ids:
                for _ in range(32):
                    mid=(low+high)/2
                    if query(mid):high=mid
                    else:low=mid
                return high,query(high),queries
            low=high
        return b.copy(),[],queries

def segment_boxes(a,b,lo,hi):
    """Exact numerical segment/AABB distances, piecewise quadratic, no solve."""
    a,b=np.asarray(a,float),np.asarray(b,float);v=b-a;n=len(lo);best=np.full(n,np.inf)
    breaks=np.zeros((n,8));breaks[:,1]=1
    for j in range(3):
        if v[j]!=0:
            for k,face in enumerate((lo,hi)):breaks[:,2+2*j+k]=np.clip((face[:,j]-a[j])/v[j],0,1)
    breaks.sort(axis=1)
    def at(t):return np.sum(np.maximum(np.maximum(lo-(a+t[:,None]*v),(a+t[:,None]*v)-hi),0.)**2,axis=1)
    for k in range(8):best=np.minimum(best,at(breaks[:,k]))
    for k in range(7):
        l,r=breaks[:,k],breaks[:,k+1];mid=a+((l+r)/2)[:,None]*v;active=(mid<lo)|(mid>hi);bound=np.where(mid<lo,lo,hi)
        denominator=np.sum(active*v*v,axis=1);numerator=np.sum(active*v*(a-bound),axis=1)
        t=np.divide(-numerator,denominator,out=((l+r)/2).copy(),where=denominator>0);best=np.minimum(best,at(np.clip(t,l,r)))
    return best

def envelopes(slot,params):
    p=np.array(slot['position'],float);need(slot['quaternion']==[0.,0.,0.,1.],'Only frozen identity orientation admitted')
    if slot['instrument']=='cusa':
        need(params['radiusMm']==1.5 and params['contactGeometryRevision']==2,'Frozen CUSA geometry');return [(p+[0,0,-40],p.copy(),.5,'shaft'),(p.copy(),p.copy(),1.5,'working')]
    need(slot['instrument']=='brain_retractor','Unsupported catalog tool');need(params['radiusMm']==3. and params['depthMm']==2.,'Frozen retractor geometry');radius=params['radiusMm'];depth=min(params['depthMm'],5.)
    return [(p+[0,0,-40],p+[0,0,-2],.5,'shaft'),(p+[-radius,0,0],p+[radius,0,0],.4,'working'),(p+[-radius,0,-depth],p+[radius,0,-depth],.4,'working')]

def operator_identity(g,alpha,times,removed):
    h=hashlib.sha256(b'resectionlab.exterior-source-system.v1')
    for v in [*(g[k] for k in KEYS),alpha,times,g['coreFreeDofs'],np.unique(np.asarray(removed,np.int64)),np.empty((0,2),np.int64)]:
        a=np.ascontiguousarray(v);h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()

def contacts_from_holds(holds,slot_stiffness):
    result=[]
    for slot,h in sorted(holds.items()):
        need('target' in h and h['cells'],'Unsupported rigid/empty hold');cells=sorted(set(h['cells']))
        for c in cells:
            t=np.subtract(h['target'],h['origin'])+np.asarray(h.get('baseDisplacements',{}).get(str(c),[0.,0.,0.]))
            result.append({'slot':slot,'sourceCell':c,'stiffnessNPerMm':slot_stiffness[slot]/len(cells),'targetDisplacementMm':t.tolist()})
    return result

def state_hash(u,z,values,ledger,target,exposure):
    h=hashlib.sha256(field_hash(u,z).encode());h.update(json.dumps(values,sort_keys=True,allow_nan=False).encode());h.update(json.dumps(ledger,sort_keys=True,allow_nan=False).encode());h.update(np.asarray(target,np.float64).tobytes());h.update(np.asarray(exposure,np.float64).tobytes());return h.hexdigest()

class Audit:
    def __init__(self,directory,protocol):
        self.start=time.perf_counter();self.directory=Path(directory).resolve();self.protocol=Path(protocol).resolve();self.p=read(protocol);self.report=read(self.directory/'report.json');self.cache={};self.geometries={};self.checked=[];self.local_cache={}
        self.deadline=self.start+30.;self.errors={};scope(self.p);scope(self.report);need(self.directory.is_relative_to(ROOT/'testcases'),'Evidence outside testcases')
        need(self.report['status']=='completed' and self.report['protocolSha256']==sha(self.protocol) and self.report['sourceInputsUnchanged'] is True,'Report/protocol incomplete')
        need(self.p['caps']==dict(wallSeconds=180,rssBytes=3221225472,outputBytes=33554432,numericalThreads=1,retries=0) and self.p['workerWallSeconds']==145 and self.p['checkerWallSeconds']==30 and self.p['checkerCaps']['wallSeconds']==30,'Changed workflow/checker caps')
        need(self.report['wallSeconds']<=145 and self.report['peakRssBytes']<=3221225472 and all(v=='1' for v in self.report['numericalThreads'].values()),'Worker resource/thread receipt')
        need('DEVELOPMENT' in self.p['sourceRole'] and self.report['sourceRole']==self.p['sourceRole'],'Source role')
        need(self.p['passes']==2 and self.p['expectedTicks']==10 and self.p['expectedFixedSteps']==120 and self.p['expectedSamples']==20,'Pilot coverage declaration')
        need(self.p['algebraicCorePolicy']==self.report['algebraicCorePolicy']=='contact-core-fresh','Algebraic policy')
        profile=read(self.bound(self.p['costProfileReport']));accepted=read(self.bound(self.p['costProfileAcceptance']))
        need(profile['status']=='completed' and profile['sourceInputsUnchanged'] and accepted['sourceCostProfileComplete'] and accepted['supervisionStatus']=='completed' and profile['selectedFastestPolicy']=='contact-core-fresh','Cost profile admission')
        chosen=[v for v in profile['variants'] if v['policy']=='contact-core-fresh'];need(len(chosen)==1 and chosen[0]['rootCoreFreeDofsSha256']==self.p['rootCoreFreeDofsSha256'],'Measured core identity')
        self.limits=self.p['limits'];self.energy_tol=self.limits['energyNmm'];self.force_tol=self.limits['forceN'];self.balance_tol=self.limits['balanceNmm']
        need(self.limits==dict(forceN=1e-10,energyNmm=1e-10,balanceNmm=1e-8,maxGradientNorm=.2,maxCenterDisplacementMm=3.,maxReactionN=.08),'Changed numerical/physical budgets')
        need(self.bound(self.p['independentChecker']).resolve()==Path(__file__).resolve(),'Checker identity')
        self.q=runpy.run_path(str(self.bound(self.p['independentQuadrature'])))
        for path,digest in self.report['sourceHashes'].items():need(sha(self.rootpath(path))==digest,'Worker source changed '+path)
        launch_path=self.directory.parent.with_name(self.directory.parent.name+'-launch.json')
        launch=read(launch_path);need(all(launch[k]==self.report[k] for k in ('sourceHashes','sourceCommit') if k in launch),'Launch/source mismatch')
        need(launch['commit']==self.report['sourceCommit'] and launch['sourceHashes']==self.report['sourceHashes'] and launch['protocolSha256']==sha(self.protocol),'Launch identity mismatch')
        self.labels,self.affine,self.soft,self.target_labels,self.rate=self.case_labels(self.bound(self.p['nativeCase']))
        self.root=self.raw_geometry(self.bound(self.p['geometry']['reference']));self.alpha=np.asarray(self.root.pop('alpha'),float);self.times=np.asarray(self.root.pop('times'),float);self.common=np.asarray(self.root.pop('commonCells'))
        need(self.alpha.ndim==1 and 0<len(self.alpha)<=2 and self.times.shape==self.alpha.shape and np.isfinite(self.alpha).all() and np.all(self.alpha>=0) and np.all(self.times>0),'Spectrum')
        self.gradients,self.weight=self.q['quadrature'](self.affine);self.volume=abs(float(np.linalg.det(self.affine[:3,:3])))
        self.root_lookup={int(c):i for i,c in enumerate(self.root['cells'])};self.root_free=np.flatnonzero(self.root['freeDofs'])
    def time(self):need(time.perf_counter()<self.deadline,'Checker cooperative30s cap')
    def rootpath(self,path):
        p=(ROOT/path).resolve();need(p.is_relative_to(ROOT),'Input outside source workspace');return p
    def bound(self,entry):
        p=self.rootpath(entry['path']);need(sha(p)==entry['sha256'],'Frozen input hash '+str(p));return p
    def exported(self,name,digest):
        p=(self.directory/name).resolve();need(p.is_relative_to(self.directory) and p!=self.directory,'Export path escape');need(sha(p)==digest,'Export SHA '+name);return p
    def npz(self,path):
        key=str(path)
        if key not in self.cache:
            with np.load(path,allow_pickle=False) as f:self.cache[key]={k:f[k] for k in f.files}
            need(all(not a.dtype.hasobject for a in self.cache[key].values()),'Object array export')
        return self.cache[key]
    def raw_geometry(self,path):
        d=self.npz(path);return {k:d[k] for k in KEYS+('coreFreeDofs','alpha','times','commonCells') if k in d}
    def case_labels(self,path):
        with zipfile.ZipFile(path) as archive:
            manifest=json.loads(archive.read('manifest.json'));need(manifest['frame']=='RAS+','Source frame')
            # Lazy nested NPZ access: never request MRI/image/source arrays.
            payload=archive.read('arrays.npz');need(hashlib.sha256(payload).hexdigest()==manifest['array_sha256'],'Native categorical container SHA')
            with np.load(io.BytesIO(payload),allow_pickle=False) as f:
                affine=finite(f['affine'],(4,4),'source affine');mapping=manifest['artifacts']['surgical_toolbox']['mapping'];index=manifest['array_index']['compartments'];labels=None;soft=set();target=[]
                for region in mapping:
                    mask=np.asarray(f[index[region['name']]]);need(mask.ndim==3 and mask.dtype.kind in 'biu','Source categorical mask')
                    if labels is None:labels=np.zeros(mask.shape,np.uint16)
                    need(mask.shape==labels.shape and type(region['id']) is int and region['id']>0,'Source mapping')
                    labels[mask.astype(bool)]=region['id']
                    if region['material'] in ('soft_tissue','tumor'):soft.add(region['id'])
                    if region.get('target',False):target.append(region['id'])
        props=manifest['metadata'].get('surgical_properties',{});profile=props.get('cusaV3');self.handle_stiffness=props.get('brainContinuumV1',{}).get('handleStiffnessNPerMm')
        self.materials={r['id']:r['material'] for r in mapping};self.blocking={i for i,m in self.materials.items() if m in ('bone','vessel','unknown')}
        rate=2. if profile is None else scalar(profile['capacityMm3PerSecond'],'rate')
        need(rate==2.,'Frozen source CUSA rate changed');return labels,affine,soft,sorted(target),rate
    def geometry(self,entry):
        path=self.exported(entry['file'],entry['sha256'])
        if str(path) in self.geometries:return self.geometries[str(path)]
        g=self.raw_geometry(path);self.validate_geometry(g);self.geometries[str(path)]=g;return g
    def validate_geometry(self,g):
        self.time();cells=g['cells'];en=g['elementNodes'];nodes=g['nodes'];free=g['freeDofs'];n=len(nodes)
        need(cells.ndim==1 and cells.dtype.kind in 'iu' and len(cells)==len(set(map(int,cells))) and 0<len(cells)<=16000,'Source owner support')
        need(np.all((cells>=0)&(cells<self.labels.size)) and np.array_equal(np.sort(cells),cells),'Source owner range/order')
        need(g['shape'].dtype.kind in 'iu' and np.array_equal(g['shape'],self.labels.shape) and np.array_equal(g['affine'],self.affine),'Source geometry frame')
        need(en.shape==(len(cells),8) and en.dtype.kind in 'iu' and np.all((en>=0)&(en<n)) and len(np.unique(en))==n,'Element graph')
        need(nodes.shape==(n,3) and free.shape==(n*3,) and free.dtype.kind=='b','Node/free support')
        need(g['muPa'].shape==cells.shape and g['nu'].shape==cells.shape and np.all(g['muPa']>0) and np.all((g['nu']>=0)&(g['nu']<=.45)),'Material bounds')
        need(all(int(self.labels.flat[c]) in self.soft for c in cells),'Nonsoft mechanical source owner')
        xyz=np.array(np.unravel_index(cells,self.labels.shape)).T;expected=(xyz[:,None,:]+CORNERS-.5)@self.affine[:3,:3].T+self.affine[:3,3]
        near(nodes[en],expected,1e-10,'Source corner coordinates');lookup=self.root_lookup
        need(all(int(c) in lookup for c in cells),'Activated source owner');rows=np.array([lookup[int(c)] for c in cells],int)
        need(np.array_equal(g['muPa'],self.root['muPa'][rows]) and np.array_equal(g['nu'],self.root['nu'][rows]),'Changed source material')
        ancestors=self.root['elementNodes'][rows].ravel();targets=en.ravel();parents=np.full(n,len(self.root['nodes']),np.int64);np.minimum.at(parents,targets,ancestors)
        need(np.array_equal(ancestors,parents[targets]),'Joined root source sectors')
        line=(3*parents[:,None]+np.arange(3)).ravel();need(np.array_equal(free,self.root['freeDofs'][line]),'Changed inherited support')
        lineage=np.searchsorted(self.root_free,line[free]);need(np.array_equal(self.root_free[lineage],line[free]),'Free source lineage')
        mask=np.isin(lineage,self.root_core);core=np.flatnonzero(mask);rank={int(v):i for i,v in enumerate(self.root_core)};core=core[np.argsort([rank[int(lineage[i])] for i in core],kind='stable')]
        need(g['coreFreeDofs'].ndim==1 and g['coreFreeDofs'].dtype.kind in 'iu' and np.array_equal(g['coreFreeDofs'],core),'Root contact partition changed')
        # Without cuts, incident owner sectors are exactly face-connected components.
        by_parent={}
        for row in range(len(cells)):
            for local,node in enumerate(en[row]):by_parent.setdefault(int(parents[node]),{})[tuple(xyz[row])]=int(node)
        for owners in by_parent.values():
            pending=set(owners);components=[]
            while pending:
                a=pending.pop();group={a};stack=[a]
                while stack:
                    a=stack.pop()
                    for axis in range(3):
                        for sign in (-1,1):
                            b=list(a);b[axis]+=sign;b=tuple(b)
                            if b in pending:pending.remove(b);group.add(b);stack.append(b)
                ids={owners[a] for a in group};need(len(ids)==1,'Artificial uncut node split');components.append(next(iter(ids)))
            need(len(set(components))==len(components),'Healed/disconnected source node join')
        g['_lookup']={int(c):i for i,c in enumerate(cells)};g['_parents']=parents
    def root_partition(self,contacts):
        full=set()
        for c in contacts:
            need(c['sourceCell'] in self.root_lookup,'Root contact owner');nodes=self.root['elementNodes'][self.root_lookup[c['sourceCell']]]
            for node in nodes:
                for axis in range(3):
                    dof=3*int(node)+axis
                    if self.root['freeDofs'][dof]:full.add(dof)
        self.root_core=np.searchsorted(self.root_free,sorted(full));need(0<len(self.root_core)<len(self.root_free),'Root contact core')
        digest=hashlib.sha256(self.root_core.astype(np.int64).tobytes()).hexdigest();need(digest==self.p['rootCoreFreeDofsSha256']==self.report['rootCoreFreeDofsSha256'],'Root core hash')
    def fields(self,geometry,filename,digest,names=('displacement','memories')):
        g=geometry;d=self.npz(self.exported(filename,digest));free=np.flatnonzero(g['freeDofs']);need(d['freeIndices'].dtype.kind in 'iu' and np.array_equal(d['freeIndices'],free),'Field support order')
        u=np.zeros(len(g['nodes'])*3);z=np.zeros((len(self.alpha),len(u)));u[free]=finite(d[names[0]],(len(free),),names[0]);z[:,free]=finite(d[names[1]],(len(self.alpha),len(free)),names[1]);return u.reshape(-1,3),z.reshape(len(self.alpha),-1,3),d
    def spring_data(self,g,contacts,target):
        rows=[g['_lookup'][c['sourceCell']] for c in contacts];cn=g['elementNodes'][rows] if rows else np.empty((0,8),int);k=np.repeat([c['stiffnessNPerMm'] for c in contacts],3);t=np.array([c['targetDisplacementMm'] for c in contacts],float).ravel();near(target,t,0.,'Spring target');return cn,k,t
    def local(self,g,u,z):
        key=(id(g),field_hash(u,z))
        if key in self.local_cache:return self.local_cache[key]
        ku,eu=self.q['elastic_fields'](u[g['elementNodes']],g['muPa'],g['nu'],self.gradients,self.weight);force=ku.copy();energy=eu.copy();memory=[]
        for a,v in zip(self.alpha,z):
            kv,ev=self.q['elastic_fields'](v[g['elementNodes']],g['muPa'],g['nu'],self.gradients,self.weight);force+=a*(ku-kv);energy+=.5*a*np.sum((u-v)[g['elementNodes']]*(ku-kv),axis=(1,2));memory.append(kv)
        result=(force,energy,ku,memory)
        if len(self.local_cache)>45:self.local_cache.pop(next(iter(self.local_cache)))
        self.local_cache[key]=result;return result
    def potential(self,g,u,contacts,target):
        cn,k,t=self.spring_data(g,contacts,target);e=u[cn].mean(axis=1).ravel()-t;return float(.5*e@(k*e)),cn,k,e
    def transfer(self,old,new,u,z):
        oldrows=np.array([old['_lookup'][int(c)] for c in new['cells']]);source=old['elementNodes'][oldrows].ravel();target=new['elementNodes'].ravel();n=len(new['nodes']);first=np.full(n,len(source),int);np.minimum.at(first,target,np.arange(len(target)))
        need(np.all(first<len(source)),'Unowned new node');a=u[source[first]].copy();b=z[:,source[first]].copy();bits(a[target],u[source],'Transfer displacement continuity');bits(b[:,target],z[:,source],'Transfer memory continuity')
        need(np.all(a.ravel()[~new['freeDofs']]==0) and np.all(b.reshape(len(b),-1)[:,~new['freeDofs']]==0),'New constrained history');return a,b
    def sampled_step(self,g,u,z,before,oldz,target,previous,contacts,row,h,removed):
        self.time();near(h,self.p['stepSeconds'],0.,'Sample h');need(h>0,'Nonpositive sample h')
        old_center=before[g['elementNodes']].mean(axis=1);center=u[g['elementNodes']].mean(axis=1);rest=np.array(np.unravel_index(g['cells'],self.labels.shape)).T@self.affine[:3,:3].T+self.affine[:3,3];inverse=np.linalg.inv(self.affine)
        for fraction in (.25,.5,.75,1.):
            point=rest+old_center+fraction*(center-old_center);voxel=np.rint(point@inverse[:3,:3].T+inverse[:3,3]).astype(int);need(np.all((voxel>=0)&(voxel<np.array(self.labels.shape))),'Sample BE obstacle FOV')
            ids=np.ravel_multi_index(voxel.T,self.labels.shape);need(not np.any(np.isin(self.labels.ravel()[ids],list(self.blocking))&~np.isin(ids,removed)),'Sample BE obstacle increment')
        bits(target,previous,'Fixed binding target must be stationary');near(z,(self.times/(self.times+h))[:,None,None]*oldz+(h/(self.times+h))[:,None,None]*u,1e-9,'Memory update')
        force,energy,ku,kz=self.local(g,u,z);_,oldenergy,kb,kzb=self.local(g,before,oldz);en=g['elementNodes']
        du=(u-before)[en];dk=ku-kb;num=.5*np.sum(du*dk);loss=0.
        for j,(a,tau) in enumerate(zip(self.alpha,self.times)):
            dz=(z[j]-oldz[j])[en];dzk=kz[j]-kzb[j];dv=du-dz
            loss+=a*tau/h*np.sum(dz*dzk);num+=.5*a*np.sum(dv*(dk-dzk))
        contact,cn,k,e=self.potential(g,u,contacts,target);oldcontact,_,_,oe=self.potential(g,before,contacts,previous)
        num+=.5*(e-oe)@(k*(e-oe));reaction=-k*e;work=float(reaction@(target-previous));total=float(energy.sum()+contact);old=float(oldenergy.sum()+oldcontact)
        assembled=np.zeros_like(u);np.add.at(assembled,en,force)
        for nodes,f in zip(cn,(k*e).reshape(-1,3)):np.add.at(assembled,nodes,f/8)
        residual=float(np.max(np.abs(assembled.ravel()[g['freeDofs']]),initial=0.));need(residual<=self.force_tol,'Independent Q1 free equilibrium')
        balance=work-(total-old)-loss-num
        for name,value in [('energyNmm',total),('oldEnergyNmm',old),('contactEnergyNmm',contact),('dissipatedEnergyNmm',loss),('numericalDissipationNmm',num),('workNmm',work),('targetWorkNmm',work),('bodyWorkNmm',0.),('energyBalanceResidualNmm',balance)]:near(row[name],value,self.energy_tol,'Independent '+name)
        near(row['contactReactionN'],reaction,self.force_tol,'Independent reaction');near(row['maxFreeResidualN'],residual,self.force_tol,'Independent residual statistic');need(abs(balance)<=self.balance_tol and loss>=-1e-12 and num>=-1e-12,'Independent BE energy sign/balance')
        lookup=g['_lookup'];probe=dict(g,alpha=self.alpha,commonRows=np.asarray([lookup[int(c)] for c in self.common if int(c) in lookup],int))
        rates,gradient,center=self.q['field_probe_metrics'](probe,u,z);near(row['rates'][0],rates[0],1e-9,'Strain rate');near(row['rates'][1],rates[1],.001,'Stress rate');near(row['maxGradientNorm'],gradient,1e-10,'Gradient bound');near(row['maxCenterDisplacementMm'],center,1e-10,'Center bound')
        self.checked.append({'fixedStep':row['fixedStep'],'freeResidualN':residual,'energyBalanceResidualNmm':balance});return total
    def collision(self,g,u,removed,slots,params,holds,key,end=None):
        self.time();raster=Raster(self.labels,self.affine,g,u,removed,self.soft);local=g['nodes'][g['elementNodes']]+u[g['elementNodes']];lo=local.min(axis=1);hi=local.max(axis=1);present=set(map(int,g['cells']));cleared=set(removed)
        # Source affine is axis aligned (possibly permuted/signed). No patient MRI.
        A=self.affine[:3,:3];need(np.all(np.sum(np.abs(A)>1e-12,axis=0)==1) and np.all(np.sum(np.abs(A)>1e-12,axis=1)==1),'Non-axis-aligned source grid unsupported')
        inverse=raster.inverse;pieces={};shaft_proofs=0
        for slot,value in slots.items():
            spec=copy.deepcopy(value);inflation=0.
            if slot==key and end is not None:
                need(end['quaternion']==value['quaternion']==[0.,0.,0.,1.],'Swept orientation');a=np.array(value['position']);b=np.array(end['position']);spec['position']=((a+b)/2).tolist();inflation=float(np.linalg.norm(b-a))/2
            pieces[slot]=[(a,b,r+inflation,part) for a,b,r,part in envelopes(spec,params[slot])]
            own=set(holds.get(slot,{}).get('cells',[]))
            for a,b,radius,part in pieces[slot]:
                distance=segment_boxes(a,b,lo,hi);possible={int(g['cells'][i]) for i in np.flatnonzero(distance<=(radius+EPS)**2)}
                # Complete resting source cubes outside the local field and raster destinations.
                bounds=np.array(list(product(*zip(np.minimum(a,b)-radius-1.,np.maximum(a,b)+radius+1.))))
                vox=bounds@inverse[:3,:3].T+inverse[:3,3];lower=np.maximum(0,np.floor(vox.min(axis=0)-.5).astype(int));upper=np.minimum(np.array(self.labels.shape)-1,np.ceil(vox.max(axis=0)+.5).astype(int))
                rasterowners=set()
                if np.all(lower<=upper):
                    coords=np.stack(np.meshgrid(*[np.arange(l,h+1) for l,h in zip(lower,upper)],indexing='ij'),axis=-1).reshape(-1,3)
                    need(len(coords)<100000,'Collision candidate budget');ids=np.ravel_multi_index(coords.T,self.labels.shape);centers=coords@A.T+self.affine[:3,3];half=np.sum(np.abs(A),axis=1)/2
                    touched=ids[segment_boxes(a,b,centers-half,centers+half)<=(radius+EPS)**2]
                    for owner in touched:
                        owner=int(owner)
                        if owner in raster.members:rasterowners.update(raster.members[owner])
                        elif owner not in raster.cleared and self.labels.flat[owner]:rasterowners.add(owner)
                        if self.labels.flat[owner] and owner not in present and owner not in cleared:possible.add(owner)
                if part=='shaft':need(not ((possible|rasterowners)-own),'Independent full-source shaft separation failed');shaft_proofs+=1
                else:need(all(int(self.labels.flat[c]) in self.soft for c in (possible|rasterowners)-own),'Working material outside declared compatibility')
        def point_seg(p,a,b):
            v=b-a;n=float(v@v);t=np.clip(float((p-a)@v)/n,0.,1.) if n else 0.;return float(np.linalg.norm(p-a-t*v))
        def segment_seg(a,b,c,d):
            v=b-a;w=d-c;x=a-c;aa=float(v@v);bb=float(v@w);cc=float(w@w);dd=float(v@x);ee=float(w@x);den=aa*cc-bb*bb
            values=[point_seg(a,c,d),point_seg(b,c,d),point_seg(c,a,b),point_seg(d,a,b)]
            if den>1e-18:
                s=(bb*ee-cc*dd)/den;t=(aa*ee-bb*dd)/den
                if 0<=s<=1 and 0<=t<=1:values.append(float(np.linalg.norm(a+s*v-c-t*w)))
            return min(values)
        names=list(pieces)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                for x,y,r,_ in pieces[a]:
                    for v,w,s,_ in pieces[b]:need(segment_seg(x,y,v,w)>r+s+EPS,'Independent tool-pair cover separation')
        return shaft_proofs
    def seed_check(self):
        full_path=self.bound(self.p['fullProgramReport']);full=read(full_path)
        samples=read(self.bound({'path':str((full_path.parent/'samples.json').relative_to(ROOT)),'sha256':full['samplesSha256']}));by_stage={s['tissueState']:s['geometry'] for s in samples}
        metric_path=self.bound({'path':str((full_path.parent/full['metricsFile']).relative_to(ROOT)),'sha256':full['metricsSha256']});metrics=json.loads(gzip.decompress(metric_path.read_bytes()));name=self.p['seed']['branch'];step=self.p['seed']['coupledStep']
        rows=[r for r in metrics if (r['branch'],r['step'])==(name,step)];need(len(rows)==1,'Seed row unique');row=rows[0];sample=next(s for s in samples if (s['branch'],s['step'])==(name,step));need(row['withinTick']==0 and row['event'] is not None,'Seed phase')
        capture_path=self.bound(self.p['preEffectCaptureReport']);capture=read(capture_path);need(capture['status']=='completed' and capture['nativeAllPredictionsExact'],'Capture completeness');frames=json.loads(gzip.decompress(self.bound({'path':str((capture_path.parent/capture['framesFile']).relative_to(ROOT)),'sha256':capture['framesSha256']}).read_bytes()));matches=[f for f in frames if (f['branch'],f['coupledStep'])==(name,step)];need(len(matches)==1,'Seed frame unique');frame=matches[0]
        need(frame['toolMechanicsRevision']==2 and frame['previousArrayHash']==row['previousArrayHash']==self.report['seed']['previousArrayHash'] and not frame['protectedSourceIds'] and not frame['guardReport']['configured'] and not frame['cutFaces'] and not frame['detachedFragments'],'Seed context/field binding')
        for key in ('actionIndex','operation','slot'):need(frame[key]==row[key],'Seed action phase')
        native=json.loads(gzip.decompress(self.bound(self.p['nativeInput']).read_bytes()));prefix=sum(t['actionIndex']<self.p['commonActions'] for t in native['ticks']);index=frame['nativeTickIndex'];need(index==prefix+row['suffixTick'] and frame['mechanics']==native['ticks'][index],'Seed native clock index')
        need(frame['removedSourceIds']==native['topologies'][native['ticks'][index-1]['topologyId']]['removedSourceIds'] and frame['elapsedSecondsBefore']==native['ticks'][index-1]['timeSeconds'],'Seed old mask/time')
        old=self.raw_geometry(self.bound(by_stage[row['tissueState']-1]));current=self.raw_geometry(self.bound(by_stage[row['tissueState']]));d=self.npz(self.bound({'path':str((full_path.parent/sample['file']).relative_to(ROOT)),'sha256':sample['sha256']}));free=np.flatnonzero(current['freeDofs']);need(np.array_equal(free,d['freeIndices']),'Seed free support')
        u=np.zeros(current['nodes'].size);z=np.zeros((len(self.alpha),len(u)));u[free]=d['beforeDisplacement'];z[:,free]=d['beforeMemories'];u=u.reshape(-1,3);z=z.reshape(len(self.alpha),-1,3)
        lookup={int(c):i for i,c in enumerate(old['cells'])};rows=np.array([lookup[int(c)] for c in current['cells']]);on=old['elementNodes'][rows].ravel();nn=current['elementNodes'].ravel();before=np.full_like(old['nodes'],np.nan);memory=np.full((len(self.alpha),len(before),3),np.nan);before[on]=u[nn];memory[:,on]=z[:,nn]
        orphan=np.setdiff1d(np.arange(len(before)),np.unique(on));need(np.array_equal(orphan,d['orphanOldNodeIndices']),'Seed orphan coverage');before[orphan]=d['orphanOldDisplacement'];memory[:,orphan]=d['orphanOldMemories'];bits(before[on],u[nn],'Seed retained u');bits(memory[:,on],z[:,nn],'Seed retained memory');need(np.isfinite(before).all() and np.isfinite(memory).all() and field_hash(before,memory)==row['previousArrayHash'],'Seed complete old field hash')
        initial=self.report['initial'];need(initial['contacts']==frame['mechanics']['contacts'],'Seed captured contacts');self.root_partition(initial['contacts']);g=self.geometry(initial['geometry'])
        survey_path=self.bound(self.p['preEffectSurveyReport']);survey=read(survey_path);events=[e for e in survey['events'] if (e['branch'],e['step'])==(name,step)];need(survey['status']=='completed' and len(events)==1,'Seed prior discrepancy report')
        witness=read(self.bound({'path':str((survey_path.parent/events[0]['file']).relative_to(ROOT)),'sha256':events[0]['sha256']}));need(witness['comparison']['allCusaDecisionsExact'] is False,'Erased seed discrepancy')
        metadata=self.report['seed'];expected={'branch':name,'coupledStep':step,'nativeSeedMask':sorted(set(frame['removedSourceIds'])&set(map(int,self.root['cells']))),'rootCells':len(self.root['cells']),'rootTotalDofs':self.root['nodes'].size,'rootFreeDofs':int(self.root['freeDofs'].sum()),'seedCells':len(g['cells']),'seedTotalDofs':g['nodes'].size,'seedFreeDofs':int(g['freeDofs'].sum()),'totalRemovedSourceOwnersAtSeed':len(frame['removedSourceIds']),'elapsedSeconds':frame['elapsedSecondsBefore'],'remainderSeconds':frame['nativeRemainderBeforeSeconds'],'commands':self.p['commands'],'initialInterpretation':'Declared coupled pre-event fields plus captured native pending-work ledger; not an original20mm episode','nativeOrderAtSeed':witness['predictions']['native8']['orderedSourceIds'],'coupledOrderAtSeed':witness['predictions']['coupled20']['orderedSourceIds']}
        need(all(metadata[k]==v for k,v in expected.items()),'Seed/source/full-DOF declaration')
        pending=sum((Fraction(*v) for v in frame['cusaBefore']['progressExact'].values()),Fraction(0));params=frame['parametersBySlot'][frame['slot']];budget=self.rate*params['power']*params['duty']*self.p['tickSeconds'];bound=int((pending+self.p['cusaTicks']*Fraction.from_float(budget))//Fraction.from_float(self.volume));need(bound==self.p['maximumNewRemovals']==metadata['maximumNewRemovalsFromExactCapacity'],'Exact source-capacity removal bound')
        for key in KEYS:need(np.array_equal(g[key],old[key]),'Initial physical source geometry '+key)
        U,Z,fields=self.fields(g,initial['fieldsFile'],initial['fieldsSha256']);bits(U,before,'Initial seed u bits');bits(Z,memory,'Initial seed memory bits');need(initial['arrayHash']==row['previousArrayHash']==field_hash(U,Z),'Initial seed hash')
        for a,b in [('cusa','cusaBefore'),('holds','holds'),('slots','slots'),('parametersBySlot','parametersBySlot'),('removedSourceIds','removedSourceIds'),('elapsedSeconds','elapsedSecondsBefore'),('remainderSeconds','nativeRemainderBeforeSeconds')]:need(initial[a]==frame[b],'Initial captured '+a)
        need(all(not s['active'] for s in initial['slots'].values()) and all('target' in h for h in initial['holds'].values()),'Initial active/rigid context')
        near(fields['h'],self.p['stepSeconds'],0.,'Initial h');self.slot_stiffness={slot:scalar(self.handle_stiffness,'source handle stiffness') for slot in initial['holds']};need(contacts_from_holds(initial['holds'],self.slot_stiffness)==initial['contacts'],'Initial spring/hold contract')
        expected_target=np.array([c['targetDisplacementMm'] for c in initial['contacts']],float).ravel();bits(fields['target'],expected_target,'Initial target')
        _,energies,_,_=self.local(g,U,Z);contact,cn,k,e=self.potential(g,U,initial['contacts'],expected_target);E0=float(energies.sum()+contact);near(initial['energyNmm'],E0,self.energy_tol,'Initial nonzero energy');need(E0>0 and np.max(np.abs(Z))>0,'Nonzero initial full history')
        need(self.p['algebraicCorePolicy']=='contact-core-fresh','Contact policy');return frame,g,U,Z,E0,energies
    def run(self):
        frame,g,U,Z,E0,cellenergy=self.seed_check();p=self.p;r=self.report;initial=r['initial'];ticks=r['ticks'];need(p['cusaTicks']==7 and p['relaxTicks']==2 and len(p['commands'])==10,'Frozen microtick intervention');need(len(ticks)==p['expectedTicks'] and r['allStepReplayExact'] is True and r['allStepCount']==120 and r['sampleWorlds']==20,'Run coverage')
        for path,digest in r['inputHashes'].items():need(sha(self.rootpath(path))==digest,'Input changed '+path)
        steps=json.loads(gzip.decompress(self.exported(r['metricsFile'],r['metricsSha256']).read_bytes()));samples=read(self.exported(r['samplesFile'],r['samplesSha256']));replay=read(self.exported(r['replayFile'],r['replaySha256']))
        need(len(steps)==120 and len(samples)==20 and len(replay['records'])==120 and len(replay['ticks'])==10,'Replay/sample coverage')
        expected={(i,j,i*12+j+1) for i in range(10) for j in (0,11)};by_sample={}
        for s in samples:
            key=(s['tick'],s['withinTick'],s['fixedStep']);need(key not in by_sample,'Duplicate sample world');by_sample[key]=s
        need(set(by_sample)==expected,'First/last sample worlds changed')
        for n,(row,rr) in enumerate(zip(steps,replay['records']),1):
            need(row['fixedStep']==rr['fixedStep']==n and rr['arrayHash']==row['arrayHash'] and rr['recordSha256']==record_hash(row),'Complete step replay digest')
        need(stripped(ticks)==stripped(replay['ticks']),'Complete causal tick replay changed')
        values={k:copy.deepcopy(initial[k]) for k in ('holds','contacts','removedSourceIds','slots','parametersBySlot','cusa','elapsedSeconds','remainderSeconds')};values['removed']=values.pop('removedSourceIds')
        values.update(energyNmm=initial['energyNmm'],initialEnergyNmm=initial['energyNmm'],removedEnergyNmm=0.,contactJumpNmm=0.,workNmm=0.,lossNmm=0.,fixedSteps=0,ticks=0,sourceOperatorIdentity=operator_identity(g,self.alpha,self.times,values['removed']))
        ledger={'removedSourceIds':[],'removedEnergyNmm':0.,'events':[]};target=np.array([c['targetDisplacementMm'] for c in values['contacts']],float).ravel();exposure=np.zeros(2)
        need(state_hash(U,Z,values,ledger,target,exposure)==initial['stateHash'],'Initial complete causal state hash')
        old_mask=set(values['removed']);seed_mask=old_mask.copy();all_new=set();pos=0;shaft_proofs=0;removal_checks=[];cusa_checks=[];advanced=False
        need(len(values['removed'])==len(set(values['removed'])) and all(self.labels.flat[c] for c in values['removed']),'Whole original-source mask')
        need(set(map(int,g['cells']))==set(map(int,self.root['cells']))-old_mask,'Initial source/domain cover')
        force,_,_,_=self.local(g,U,Z);assembled=np.zeros_like(U);np.add.at(assembled,g['elementNodes'],force);_,cn,k,e=self.potential(g,U,values['contacts'],target)
        for nodes,f in zip(cn,(k*e).reshape(-1,3)):np.add.at(assembled,nodes,f/8)
        need(float(np.max(np.abs(assembled.ravel()[g['freeDofs']]),initial=0.))<=self.force_tol,'Initial Q1 equilibrium')
        for ti,tick in enumerate(ticks):
            self.time();command=p['commands'][ti];need(tick['command']==command,'Frozen physical command changed');need(tick['predictedStatePropagated'] is True and tick['physicalValidationPassed'] is False and tick['largerDomainPublicActionsAdmitted'] is False,'Tick scope')
            before_hash=state_hash(U,Z,values,ledger,target,exposure);need(tick['beforeStateHash']==before_hash,'Tick before causal state hash')
            key=command['slot'];op=command['operation'];dt=scalar(command['durationSeconds'],'tick dt');need(dt==.1 and type(command['effectiveActive']) is bool and command['effectiveActive']==(op=='apply'),'Tick activation/duration')
            need(op in ('apply','release','inspect') and not any(s['active'] for s in values['slots'].values()),'Unsupported operation/latch')
            old_g,old_u,old_z=g,U,Z;old_contacts=copy.deepcopy(values['contacts']);old_target=target.copy();old_potential,_,_,_=self.potential(old_g,old_u,old_contacts,old_target);old_energy_value=values['energyNmm'];old_cellenergy=cellenergy
            pose=values['slots'][key] if command['pose']=='current' else command['pose'];requested=finite(pose['position'],(3,),'requested pose');q=pose['quaternion'];need(q==[0.,0.,0.,1.],'Frozen source orientation')
            if op=='apply':
                need(values['slots'][key]['instrument']=='cusa' and key not in values['holds'],'CUSA-only activity');need(values['parametersBySlot'][key]['radiusMm']==1.5,'CUSA radius')
                raster=Raster(self.labels,self.affine,old_g,old_u,values['removed'],self.soft);resolved,order,queries=raster.limiter(values['slots'][key]['position'],requested,self.soft)
                expected_limiter={'startMm':values['slots'][key]['position'],'requestedMm':requested.tolist(),'radiusMm':1.5,'resolvedMm':resolved.tolist(),'retainedSourceIds':order,'queries':queries}
                need(tick['limiter']==expected_limiter and tick['orderedSourceIds']==order,'Independent current-state limiter/order trace')
                shaft_proofs+=self.collision(old_g,old_u,values['removed'],values['slots'],values['parametersBySlot'],values['holds'],key,{'position':resolved.tolist(),'quaternion':q})
                after,removed,events=exact_capacity(values['cusa'],values['removed'],order,dt,values['parametersBySlot'][key],self.volume,self.rate)
                need(after==tick['cusaAfter'] and events==tick['cusaEvents'] and removed==tick['removedSourceIds'],'Independent propagated capacity ledger')
                values['cusa']=after;values['removed']=removed
                advanced|=float(np.linalg.norm(resolved-np.array(initial['slots'][key]['position'])))>1e-8
                cusa_checks.append({'tick':ti,'orderedSourceIds':order,'inputCapacityMm3':events[0]['inputCapacityMm3'] if events else 0.,'removedSourceIds':sorted(set(removed)-old_mask)})
            else:
                bits(requested,values['slots'][key]['position'],'Stationary release/inspect');need(tick['limiter'] is None and tick['orderedSourceIds']==[] and tick['cusaEvents']==[],'Inactive CUSA effect')
                need(tick['cusaAfter']==values['cusa'] and tick['removedSourceIds']==values['removed'],'Passive ledger changed');resolved=requested
                if op=='release':need(key in values['holds'],'Release without hold');del values['holds'][key]
                shaft_proofs+=self.collision(old_g,old_u,values['removed'],values['slots'],values['parametersBySlot'],values['holds'],key,{'position':resolved.tolist(),'quaternion':q})
            need(tick['resolvedPose']=={'position':resolved.tolist(),'quaternion':q},'Resolved pose changed');values['slots'][key].update(position=resolved.tolist(),quaternion=q)
            new_mask=set(values['removed']);new=sorted(new_mask-old_mask);need(old_mask<=new_mask and not all_new.intersection(new),'Nonmonotone/double owner removal');need(set(new)<=set(map(int,self.root['cells'])),'Unmodeled source removal')
            need(not set(new).intersection(c for h in values['holds'].values() for c in h['cells']),'Held owner removal unsupported');all_new.update(new)
            g=self.geometry(tick['geometry']);need(set(map(int,g['cells']))==set(map(int,self.root['cells']))-new_mask,'Dynamic source domain mask')
            U,Z=self.transfer(old_g,g,old_u,old_z);contacts=contacts_from_holds(values['holds'],self.slot_stiffness);need(tick['contacts']==contacts,'Dynamic/released contacts');target=np.array([c['targetDisplacementMm'] for c in contacts],float).ravel()
            _,transferred_energy,_,_=self.local(g,U,Z);event=tick['sourceEvent']
            if new:
                need(event is not None and event['removedSourceIds']==new and event['removedElementCount']==len(new),'Source event owner count')
                R=float(sum(old_cellenergy[old_g['_lookup'][c]] for c in new));near(event['removedStoredEnergyNmm'],R,self.energy_tol,'Removed full tissue/history energy');near(event['oldTissueEnergyNmm'],old_cellenergy.sum(),self.energy_tol,'Old tissue event energy');near(event['transferredTissueEnergyNmm'],transferred_energy.sum(),self.energy_tol,'Transferred tissue event energy')
                b=float(transferred_energy.sum()-old_cellenergy.sum()+R);near(event['tissueEventBalanceResidualNmm'],b,self.energy_tol,'Tissue event balance');need(abs(b)<=self.energy_tol,'Independent event conservation')
                oldrows=np.array([old_g['_lookup'][int(c)] for c in g['cells']]);source=old_g['elementNodes'][oldrows].ravel();newnodes=g['elementNodes'].ravel();anc={};desc={}
                for parent,child in zip(source,newnodes):anc.setdefault(int(child),set()).add(int(parent));desc.setdefault(int(parent),set()).add(int(child))
                need(event['splitParents']==sum(len(x)>1 for x in desc.values()) and event['joinedNodes']==sum(len(x)>1 for x in anc.values()),'Source split/join counts')
                values['removedEnergyNmm']+=event['removedStoredEnergyNmm'];ledger['removedEnergyNmm']+=event['removedStoredEnergyNmm'];ledger['removedSourceIds']=sorted(all_new);ledger['events'].append(copy.deepcopy(event));removal_checks.append({'tick':ti,'owners':new,'removedEnergyNmm':R})
            else:need(event is None,'Unexpected source event')
            new_potential,_,_,_=self.potential(g,U,contacts,target);J=new_potential-old_potential;near(tick['contactModelEnergyJumpNmm'],J,self.energy_tol,'Actual spring-model jump');values['contactJumpNmm']+=tick['contactModelEnergyJumpNmm'];values['contacts']=contacts;values['sourceOperatorIdentity']=operator_identity(g,self.alpha,self.times,values['removed'])
            before_step_energy=old_energy_value-(0. if event is None else event['removedStoredEnergyNmm'])+tick['contactModelEnergyJumpNmm'];steps_count=int(np.floor((values['remainderSeconds']+dt+1e-12)/p['stepSeconds']));need(steps_count==12==tick['fixedSteps'],'Physical fixed clock')
            previous_hash=field_hash(U,Z);last_U=last_Z=None
            for within in range(steps_count):
                row=steps[pos];pos+=1;need(row['tick']==ti and row['withinTick']==within and row['fixedStep']==pos and row['h']==p['stepSeconds'],'Step index/h');need(row['contacts']==contacts and row['sourceOperatorIdentity']==values['sourceOperatorIdentity'],'Step source/binding')
                need(row['previousArrayHash']==previous_hash,'Complete full-array hash chain');need(len(row['arrayHash'])==64,'Invalid full-array digest');bytes.fromhex(row['arrayHash'])
                near(row['oldEnergyNmm'],before_step_energy,self.energy_tol,'Event-to-step old energy');body=scalar(row['bodyWorkNmm'],'body work');work=scalar(row['workNmm'],'work');near(body,0.,self.energy_tol,'Fixed zero body work');near(row['targetWorkNmm'],0.,self.energy_tol,'Fixed target work');near(work,0.,self.energy_tol,'Fixed-binding work')
                energy=scalar(row['energyNmm'],'energy');need(energy>=-1e-12,'Negative stored energy');diss=scalar(row['dissipatedEnergyNmm'],'loss');num=scalar(row['numericalDissipationNmm'],'BE loss');need(diss>=-1e-12 and num>=-1e-12,'Negative step loss');balance=work-(energy-before_step_energy)-diss-num;near(row['energyBalanceResidualNmm'],balance,self.energy_tol,'Scalar BE identity');need(abs(balance)<=self.balance_tol,'BE balance bound')
                residual=scalar(row['maxFreeResidualN'],'free residual');gradient=scalar(row['maxGradientNorm'],'gradient');center=scalar(row['maxCenterDisplacementMm'],'center');need(0<=residual<=self.force_tol and 0<=gradient<=p['limits']['maxGradientNorm'] and 0<=center<=p['limits']['maxCenterDisplacementMm'],'Step force/domain bound')
                reaction=finite(row['contactReactionN'],(3*len(contacts),),'reaction');spring_values=np.repeat([c['stiffnessNPerMm'] for c in contacts],3);expected_contact=float(.5*np.sum(reaction*reaction/spring_values)) if len(contacts) else 0.;near(row['contactEnergyNmm'],expected_contact,self.energy_tol,'Allstep contact potential/reaction');need(-1e-12<=row['contactEnergyNmm']<=energy+self.energy_tol,'Contact versus total potential');forces={}
                for c,f in zip(contacts,reaction.reshape(-1,3)):forces[c['slot']]=forces.get(c['slot'],np.zeros(3))+f
                need(all(np.linalg.norm(v)<=p['limits']['maxReactionN'] for v in forces.values()),'Handle force bound')
                rates=finite(row['rates'],(2,),'rates');need(np.all(rates>=0),'Negative exposure rate');exposure+=rates*p['stepSeconds'];values['workNmm']+=work;values['lossNmm']+=diss+num;values['energyNmm']=energy;values['fixedSteps']+=1
                cumulative=energy-values['initialEnergyNmm']+values['removedEnergyNmm']-values['contactJumpNmm']+values['lossNmm']-values['workNmm'];near(row['cumulativeEventBalanceResidualNmm'],cumulative,self.energy_tol,'Cumulative event energy');need(abs(cumulative)<=self.balance_tol,'Cumulative balance bound')
                if within in (0,11):
                    s=by_sample[(ti,within,pos)];need(s['geometry']==tick['geometry'] and s['contacts']==contacts and s['sourceOperatorIdentity']==values['sourceOperatorIdentity'],'Sample source/time binding')
                    after_u,after_z,d=self.fields(g,s['fieldsFile'],s['fieldsSha256']);before_u,before_z,_=self.fields(g,s['fieldsFile'],s['fieldsSha256'],('beforeDisplacement','beforeMemories'))
                    need(field_hash(after_u,after_z)==row['arrayHash'] and field_hash(before_u,before_z)==row['previousArrayHash'],'Sample full-field hashes');near(d['h'],p['stepSeconds'],0.,'Sample h');bits(d['target'],target,'Sample target');bits(d['previousTarget'],target,'Sample previous target')
                    if within==0:bits(before_u,U,'Exact transferred first-step u');bits(before_z,Z,'Exact transferred first-step z')
                    self.sampled_step(g,after_u,after_z,before_u,before_z,d['target'],d['previousTarget'],contacts,row,float(d['h']),values['removed'])
                    last_U,last_Z=after_u,after_z
                previous_hash=row['arrayHash'];before_step_energy=energy
            U,Z=last_U,last_Z;need(field_hash(U,Z)==tick['arrayHash']==previous_hash,'Tick final full field');_,cellenergy,_,_=self.local(g,U,Z)
            values['elapsedSeconds']+=dt;values['remainderSeconds']=max(0.,round(values['remainderSeconds']+dt-steps_count*p['stepSeconds'],14));values['ticks']+=1
            need(tick['elapsedSeconds']==values['elapsedSeconds'] and tick['remainderSeconds']==values['remainderSeconds'],'Elapsed/remainder drift')
            for name,value in [('energyNmm',values['energyNmm']),('cumulativeRemovedEnergyNmm',values['removedEnergyNmm']),('cumulativeContactJumpNmm',values['contactJumpNmm']),('cumulativeWorkNmm',values['workNmm']),('cumulativeLossNmm',values['lossNmm'])]:near(tick[name],value,self.energy_tol,'Tick '+name)
            near(tick['exposure'][0],exposure[0],1e-9,'Tick strain exposure');near(tick['exposure'][1],exposure[1],.001,'Tick stress exposure')
            shaft_proofs+=self.collision(g,U,values['removed'],values['slots'],values['parametersBySlot'],values['holds'],key)
            post=tick['postEndpointCovers'];catalog=[(slot,part) for slot,value in values['slots'].items() for a,b,radius,part in envelopes(value,values['parametersBySlot'][slot])]
            need([(v['slot'],v['part']) for v in post]==catalog,'Endpoint cover receipt coverage')
            for cover in post:
                need(cover['actualSweptCollisionEstablished'] is False,'Cover-to-contact promotion');sets=[]
                for name in ('rasterSourceIds','coverPointSourceIds','indeterminateCoverSourceIds'):
                    ids=cover[name];need(ids==sorted(set(ids)) and all(type(c) is int and 0<=c<self.labels.size and self.labels.flat[c] and c not in new_mask for c in ids),'Endpoint owner receipt');sets.append(set(ids))
                need(not sets[1].intersection(sets[2]),'Endpoint positive/indeterminate overlap')
                if cover['part']=='shaft':need(not (set.union(*sets)-set(values['holds'].get(cover['slot'],{}).get('cells',[]))),'Reported forbidden shaft contact')
            need(tick['afterStateHash']==state_hash(U,Z,values,ledger,target,exposure),'Tick after complete causal state hash');old_mask=new_mask
        need(pos==120 and len(self.checked)==20 and advanced,'Full intervention/advance coverage');need(not values['holds'],'Release coverage');need(p['minimumNewRemovals']<=len(all_new)<=p['maximumNewRemovals'],'Dynamic removal bound');need(np.max(np.abs(Z))>0,'Nonzero relaxed memory')
        final=r['final'];need(final['arrayHash']==field_hash(U,Z) and final['stateHash']==ticks[-1]['afterStateHash'] and final['ticks']==10 and final['fixedSteps']==120 and final['memoryNonzero'] is True,'Final state/clock coverage')
        need(final['removedSourceIds']==values['removed'] and final['newRemovedSourceIds']==sorted(all_new) and set(values['removed'])==seed_mask|all_new and final['cusa']==values['cusa'],'Final propagated source/capacity mask')
        need(final['elapsedSeconds']==values['elapsedSeconds'] and final['remainderSeconds']==values['remainderSeconds'],'Final time')
        for name in ('energyNmm','initialEnergyNmm','removedEnergyNmm','contactJumpNmm','workNmm','lossNmm'):near(final[name],values[name],self.energy_tol,'Final '+name)
        near(final['exposure'][0],exposure[0],1e-9,'Final strain exposure');near(final['exposure'][1],exposure[1],.001,'Final stress exposure');need(len(r['passes'])==2,'Fresh pass count')
        for run in r['passes']:need(run['finalStateHash']==final['stateHash'] and run['finalArrayHash']==final['arrayHash'] and run['ticks']==10 and run['fixedSteps']==120 and run['preparationReleased'] is True,'Fresh pass identity/lifetime')
        near(final['cusa']['aspiratedSolidMm3']-initial['cusa']['aspiratedSolidMm3'],len(all_new)*self.volume,1e-10,'CUSA source-volume conservation')
        source_counts={str(label):{'initialRemoved':sum(int(self.labels.flat[c])==label for c in seed_mask),'newRemoved':sum(int(self.labels.flat[c])==label for c in all_new),'finalRemoved':sum(int(self.labels.flat[c])==label for c in values['removed'])} for label in sorted(set(self.soft))}
        return {'schema':'resectionlab.closed-loop-independent.v1','status':'passed','initialInterpretation':self.report['seed']['initialInterpretation'],'sourceRole':self.p['sourceRole'],'allStepRecords':120,'sampleWorlds':20,'causalTicks':10,'sourceSolves':0,'appImports':0,'mriDecoded':False,'independentShaftProofs':shaft_proofs,'rootPhysicalCells':len(self.root['cells']),'rootPhysicalTotalDofs':self.root['nodes'].size,'rootPhysicalFreeDofs':int(self.root['freeDofs'].sum()),'rootAlgebraicCoreFreeDofs':len(self.root_core),'sourceCounts':source_counts,'dynamicRemovalChecks':removal_checks,'cusaChecks':cusa_checks,'samples':self.checked,'allIntermediateFieldsIndependentlyRecomputed':False,'unannotatedAnatomyValidated':False,'unconfiguredCriticalAnatomyValidated':False,'completeActionChecks':False,'patientMechanicsValidated':False,'physicalValidationPassed':False,'largerDomainPublicActionsAdmitted':False,'physiologicalRL':'hold','largeScaleRL':'hold','wallSeconds':time.perf_counter()-self.start}

def self_test(workdir=None):
    if workdir is None:
        workdir=Path(__file__).parent
        if not workdir.is_relative_to(ROOT/'testcases'):workdir=ROOT/'testcases/outputs/simulator-qualification-20261029/continuum-exterior-events-v1/independent-v1'/('closed-loop-checker-selftest-'+sha(__file__)[:12])
    workdir=Path(workdir).resolve();need(workdir.is_relative_to(ROOT/'testcases'),'Selftest evidence outside testcases');workdir.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();tests=[]
    # Exact budget threshold, stateful rank allocation and no input mutation.
    def ledger(pending):
        return {'schema':'resectionlab.cusa-contact.v3','progressMm3':{str(k):float(v) for k,v in pending.items()},'progressExact':{str(k):[v.numerator,v.denominator] for k,v in pending.items()},'contactSeconds':{str(k):.1 for k in pending},'inputCapacityMm3':0.,'aspiratedSolidMm3':0.,'unusedCapacityMm3':0.,'discardedCapacityMm3':0.,'effectiveActivationSeconds':0.}
    before=ledger({13:Fraction.from_float(.8)});old=copy.deepcopy(before);params={'power':1.,'duty':1.}
    a=exact_capacity(before,[],[13,22],.1,params,1.,2.);b=exact_capacity(before,[],[22,13],.1,params,1.,2.)
    need(a[1]==[13] and b[1]==[] and before==old,'Stateful capacity rank selftest');need(Fraction(*a[0]['progressExact']['22'])==Fraction(1,18014398509481984),'Binary capacity remainder selftest');tests.append('stateful-exact-capacity')
    for dt,removed in [(float(np.nextafter(.5,0.)),[]),(.5,[0]),(float(np.nextafter(.5,1.)),[0])]:need(exact_capacity(ledger({}),[],[0,1],dt,params,1.,2.)[1]==removed,'Capacity threshold selftest')
    tests.append('below-equal-above-capacity-threshold')
    labels=np.zeros((4,4,4),np.uint16);labels[1,1,1]=labels[2,1,1]=1;cells=np.array([21,37]);g={'cells':cells,'elementNodes':np.arange(16).reshape(2,8)};u=np.zeros((16,3));raster=Raster(labels,np.eye(4),g,u,[]);need(raster.ordered([1.5,1.,1.],{1})==[21,37],'Tie-by-owner selftest');u[:8,0]=-.1;shifted=Raster(labels,np.eye(4),g,u,[]);need(shifted.contacts([1.5,1.,1.])==raster.contacts([1.5,1.,1.]) and shifted.ordered([1.5,1.,1.],{1})==[37,21],'Raster blind rank selftest');tests.append('unchanged-raster-rank-flip')
    need(np.array_equal(raster.limiter([1.5,1.,1.],[3.,1.,1.],{1})[0],[1.5,1.,1.]),'Initial retained limiter selftest');tests.append('limiter-start-retained')
    boxes=np.array([[0.,0.,0.],[4.,0.,0.]]);dist=segment_boxes([-1.,.5,.5],[2.,.5,.5],boxes,boxes+1);near(dist,[0.,4.],1e-14,'Segment box selftest');tests.append('segment-box-distances')
    q=runpy.run_path(str(ROOT/'scripts/continuum_quadrature.py'));grad,w=q['quadrature'](np.eye(4));strain=np.diag([.01,.02,-.01]);corners=CORNERS-.5;field=(corners@strain.T)[None];force,energy=q['elastic_fields'](field,np.array([1000.]),np.array([.3]),grad,w);mu=.001;lam=2*mu*.3/(1-2*.3);expected=mu*np.sum(strain**2)+.5*lam*np.trace(strain)**2;near(energy,[expected],1e-18,'Affine Q1 energy selftest');near(.5*np.sum(field*force),expected,1e-18,'Q1 force-energy selftest');tests.append('analytical-affine-quadrature')
    # Choose a supported displacement and derive its balancing spring target;
    # verify the BE identity directly, without solving a response system.
    a=Audit.__new__(Audit);a.start=time.perf_counter();a.deadline=a.start+5.;a.local_cache={};a.checked=[];a.q=q;a.gradients=grad;a.weight=w;a.alpha=np.array([.5]);a.times=np.array([.5]);a.affine=np.eye(4);a.labels=np.zeros((3,3,3),np.uint16);a.labels[1,1,1]=1;a.blocking=set();a.energy_tol=1e-10;a.force_tol=1e-10;a.balance_tol=1e-8;h=1/120;a.p={'stepSeconds':h};a.common=np.array([13])
    g={'cells':np.array([13]),'elementNodes':np.arange(8)[None],'nodes':CORNERS+.5,'affine':np.eye(4),'muPa':np.array([1000.]),'nu':np.array([.3]),'freeDofs':np.zeros(24,bool),'_lookup':{13:0}};g['freeDofs'][21]=True
    new=np.zeros((8,3));new[7,0]=.01;before=np.zeros_like(new);before[7,0]=.02;oldz=np.zeros((1,8,3));oldz[0,7,0]=.003;memory=(a.times[:,None,None]*oldz+h*new)/(a.times[:,None,None]+h)
    def elastic(x):return q['elastic_fields'](x[None],g['muPa'],g['nu'],grad,w)
    ku,eu=elastic(new);stretch,es=elastic(new-memory[0]);internal=ku+.5*stretch;k=np.repeat(.05,3);t=new.mean(axis=0);t[0]+=8*internal[0,7,0]/k[0];contacts=[{'slot':'left','sourceCell':13,'stiffnessNPerMm':.05,'targetDisplacementMm':t.tolist()}]
    _,eb=elastic(before);_,ebs=elastic(before-oldz[0]);e=new.mean(axis=0)-t;oe=before.mean(axis=0)-t;total=float(eu.sum()+.5*es.sum()+.5*e@(k*e));old=float(eb.sum()+.5*ebs.sum()+.5*oe@(k*oe));_,ed=elastic(new-before);_,eds=elastic((new-memory[0])-(before-oldz[0]));_,edz=elastic(memory[0]-oldz[0]);loss=float(2*.5*.5/h*edz.sum());num=float(ed.sum()+.5*eds.sum()+.5*(e-oe)@(k*(e-oe)));reaction=-k*e;res=internal.copy()
    res[0]-=reaction/8;maximum=float(abs(res.reshape(-1)[21]));rates,gradient,center=q['field_probe_metrics'](dict(g,alpha=a.alpha,commonRows=np.array([0])),new,memory)
    row={'fixedStep':1,'energyNmm':total,'oldEnergyNmm':old,'contactEnergyNmm':float(.5*e@(k*e)),'dissipatedEnergyNmm':loss,'numericalDissipationNmm':num,'workNmm':0.,'targetWorkNmm':0.,'bodyWorkNmm':0.,'energyBalanceResidualNmm':old-total-loss-num,'contactReactionN':reaction,'maxFreeResidualN':maximum,'rates':rates,'maxGradientNorm':gradient,'maxCenterDisplacementMm':center}
    a.sampled_step(g,new,memory,before,oldz,t,t,contacts,row,h,[]);damaged=copy.deepcopy(row);damaged['dissipatedEnergyNmm']+=1e-8
    try:a.sampled_step(g,new,memory,before,oldz,t,t,contacts,damaged,h,[])
    except ValueError:pass
    else:raise ValueError('Damaged BE dissipation selftest')
    tests.append('full-Q1-Maxwell-BE-equilibrium-and-energy-corruption-refusal')
    # Canonical field hash distinguishes signed-zero/cause and replay corruption.
    need(field_hash(np.array([0.]),np.zeros((1,1)))!=field_hash(np.array([-0.]),np.zeros((1,1))),'Signed zero hash selftest');need(record_hash({'x':1})!=record_hash({'x':2}),'Record corruption selftest');tests.append('exact-field-record-hashes')
    bad={'physicalValidationPassed':True,'largerDomainPublicActionsAdmitted':False,'physiologicalRL':'hold','largeScaleRL':'hold','trainingUpdates':0,'newAcquiredDatasetBytes':0}
    try:scope(bad)
    except ValueError:pass
    else:raise ValueError('Scope promotion selftest')
    tests.append('scope-promotion-refusal')
    limited,ids,queries=raster.limiter([-2.,1.,1.],[2.,1.,1.],{1});need(len(queries)>35 and abs(limited[0]+1.)<1e-8 and ids==[21],'Bisection selftest');tests.append('full32-bisection-limiter')
    auditor=Audit.__new__(Audit);oldg={'_lookup':{21:0,37:1},'elementNodes':np.arange(16).reshape(2,8)};newg={'cells':np.array([37]),'nodes':np.zeros((8,3)),'elementNodes':np.arange(8)[None],'freeDofs':np.ones(24,bool)}
    old_u=np.arange(48,dtype=float).reshape(16,3)*.001;old_z=old_u[None]*.7;new_u,new_z=auditor.transfer(oldg,newg,old_u,old_z);bits(new_u,old_u[8:],'Retained corner transfer test');bits(new_z,old_z[:,8:],'Retained memory transfer test')
    badnew={'cells':np.array([21,37]),'nodes':np.zeros((8,3)),'elementNodes':np.tile(np.arange(8),(2,1)),'freeDofs':np.ones(24,bool)}
    try:auditor.transfer(oldg,badnew,old_u,old_z)
    except ValueError:pass
    else:raise ValueError('Unequal joined source-history selftest')
    tests.append('source-corner-transfer-and-unequal-join-refusal')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='checker-selftest-',dir=workdir) as temp:
        buf=io.BytesIO();brain=np.ones((3,3,3),bool);tumor=np.zeros_like(brain);tumor[1,1,1]=True
        np.savez_compressed(buf,affine=np.eye(4),compartments_0=brain,compartments_1=tumor,mri=np.array([{'mustNeverDecode':True}],object));payload=buf.getvalue()
        manifest={'frame':'RAS+','array_sha256':hashlib.sha256(payload).hexdigest(),'metadata':{},'array_index':{'compartments':{'brain':'compartments_0','tumor':'compartments_1'}},'artifacts':{'surgical_toolbox':{'mapping':[{'id':1,'name':'brain','material':'soft_tissue','target':False},{'id':2,'name':'tumor','material':'tumor','target':True}]}}}
        file=Path(temp)/'generated.ressectionlab'
        with zipfile.ZipFile(file,'w') as archive:archive.writestr('manifest.json',json.dumps(manifest));archive.writestr('arrays.npz',payload)
        labeled,affine,soft,target,rate=auditor.case_labels(file);need(labeled[1,1,1]==2 and labeled[0,0,0]==1 and soft=={1,2} and target==[2] and rate==2.,'Last mapped source label selftest')
    tests.append('categorical-last-wins-with-unreadable-MRI-sentinel')
    return {'status':'passed','selfTests':tests,'appImports':0,'sourceSolves':0,'sourceQueries':0,'sourceArraysRead':False,'wallSeconds':time.perf_counter()-start}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',nargs='?');parser.add_argument('protocol',nargs='?');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        result=self_test();result['checkerSha256']=sha(__file__);print(json.dumps(result));return
    need(args.directory is not None and args.protocol is not None,'RESULT_DIRECTORY PROTOCOL_PATH required');directory=Path(args.directory).resolve();output=directory/'checker-result.json';start=time.perf_counter()
    try:result=Audit(directory,args.protocol).run()
    except Exception as error:
        result={'schema':'resectionlab.closed-loop-independent.v1','status':'failed','reason':str(error),'sourceSolves':0,'appImports':0,'physicalValidationPassed':False,'largerDomainPublicActionsAdmitted':False,'physiologicalRL':'hold','largeScaleRL':'hold','wallSeconds':time.perf_counter()-start}
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result));raise SystemExit(1)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('samples','cusaChecks')}))

if __name__=='__main__':main()
