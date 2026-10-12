"""Final saved-scalar figures; no checkpoints, source images or native runtime."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import math
import os
import signal
import time

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
STARTED=time.monotonic()
signal.alarm(60)
os.environ['MPLCONFIGDIR']=str(HERE/'mpl-cache')
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

SOURCES=[]
CAP=8*1024**2
COLORS={'shared':'#576574','uniform':'#157f98','reverse':'#b75d2c','search':'#30764a'}


def read(relative):
    path=STAGE/relative
    with path.open('rb') as stream: raw=stream.read(CAP+1)
    if len(raw)>CAP: raise ValueError('Bounded saved scalar input required')
    decoded=raw
    if path.suffix=='.gz':
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream: decoded=stream.read(CAP+1)
        if len(decoded)>CAP: raise ValueError('Decoded scalar bound')
    SOURCES.append({'path':relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    return json.loads(decoded,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def evaluation(e, decisions):
    assert e['complete'] and e['audit']['accepted'] and e['local_reward_audit_accepted']
    assert math.isfinite(e['return'])
    return {'training_decisions':decisions,'return':e['return'],'success':e['success'],
        'evaluation_decisions':e['decision_count'],'immediate_STOP':e['decision_count']==1 and e['rewards']==[0.],
        'history_hash':e['history_hash'],'plan_seal':e['plan_seal']}


def optimizer_curve(cp):
    rows=cp['rollouts'];after=[r['optimizer_completed_before_update'] for r in rows[1:]]+[cp['optimizer_completed']]
    result=[{'training_decisions':0,'Adam_updates':0}]+[
        {'training_decisions':r['decisions'],'Adam_updates':a} for r,a in zip(rows,after)]
    assert all(a['training_decisions']<b['training_decisions'] and a['Adam_updates']<=b['Adam_updates']
               for a,b in zip(result,result[1:]))
    return result


def episodes(folder, count, segment):
    rows=[]
    # Exact checkpoint-completed count deliberately excludes the original reverse832 tail.
    for i in range(1,count+1):
        e=read(f'{folder}/episode-{i:04d}.json.gz')
        assert e['complete'] and e['episode']==i and 12<=e['start_absolute_step']<=16
        rows.append({'segment':segment,'episode':i,'start':e['start_absolute_step'],
            'decisions':e['costs']['policy_native_transitions'],'success':e['success'],'return':e['return']})
    return rows


def by_start(rows):
    return {str(s):{'episodes':sum(r['start']==s for r in rows),
        'successes':sum(r['start']==s and r['success'] for r in rows),
        'completed_episode_decisions':sum(r['decisions'] for r in rows if r['start']==s)} for s in range(12,17)}


def gather():
    pilot=read('pilot-01/report.json');parent=read('campaign-256-01/report.json')
    current=read('curriculum-1024-01/report.json')
    prior_reverse=read('curriculum-1024-01/reverse-seed0/curve.json')
    recovery=read('recovery-reverse1024-02/report.json');search=read('search-01/report.json')
    assert pilot['status']==parent['status']==recovery['status']==search['status']=='completed'
    assert recovery['source_unchanged'] and search['independent_audit']['accepted'] and search['local_reward_audit']['accepted']
    u=current['arms']['uniform-seed0'];r=recovery['arms']['reverse-seed0']
    assert u['status']==r['status']=='completed'
    p=pilot['arms']['frozen_encoder-seed0'];p256=parent['arms']['frozen_encoder-seed0']['checkpoints'][0]
    r512=next(c for c in prior_reverse['checkpoints'] if c['decisions']==512)
    assert r512==recovery['retained_reverse512']
    u1024=next(c for c in u['checkpoints'] if c['decisions']==1024)
    r1024=next(c for c in r['checkpoints'] if c['decisions']==1024)
    assert r['optimizer_resume_join']['all_tensors_and_optimizer_scalars_exact']
    assert r['optimizer_resume_join']['prior_optimizer_completed']==r512['optimizer_completed']
    shared=[evaluation(p['initial_evaluation'],0),evaluation(p['checkpoints'][0]['evaluation'],64),evaluation(p256['evaluation'],256)]
    upoints=shared+[evaluation(c['evaluation'],c['decisions']) for c in u['checkpoints']]
    rpoints=shared+[evaluation(r512['evaluation'],512),evaluation(r1024['evaluation'],1024)]
    ueps=episodes('curriculum-1024-01/uniform-seed0',u1024['segment_experience']['completed_episodes'],'uniform256_to1024')
    old_reps=episodes('curriculum-1024-01/reverse-seed0',r512['segment_experience']['completed_episodes'],'reverse256_to512')
    new_reps=episodes('recovery-reverse1024-02/reverse-seed0',r1024['segment_experience']['completed_episodes'],'recovery02_512_to1024')
    for rows,seg in ((ueps,u1024['segment_experience']),(old_reps,r512['segment_experience']),(new_reps,r1024['segment_experience'])):
        counts=by_start(rows)
        assert sum(x['successes'] for x in counts.values())==seg['successful_episodes']
        assert {s:x['successes'] for s,x in counts.items()}==seg['successful_episodes_by_start']
        assert sum(x['completed_episode_decisions'] for x in counts.values())+seg['unfinished_episode_decisions']==seg['decisions']
    ucurve=optimizer_curve(u1024);rcurve=optimizer_curve(r1024)
    assert rcurve[:9]==optimizer_curve(r512)
    discarded={k:recovery['recovery_admission'][k] for k in (
        'discarded_postcheckpoint_decisions_lower_bound','discarded_postcheckpoint_Adam_steps_lower_bound',
        'last_saved_clock','original_campaign_completed','fresh_RNG_and_environment','seamless_trajectory_claim')}
    discarded['enters_plotted_optimizer_lineage']=False
    return {'schema':'saved-local-goal-PPO-recovery02-curves-v1',
        'scope':'One TRAIN045; one seed; annotation-assisted SIMULATOR; frozen spatial encoder; no clinical/generalization inference',
        'result':'No full-start RL advantage over the retained successful bounded search',
        'worker_report_status':{'original_curriculum':current['status'],'recovery02':recovery['status']},
        'shared_parent_evaluations':shared,
        'arms':{'uniform':{'evaluations':upoints,'optimizer_curve':ucurve,'training_by_start':by_start(ueps),
                    'complete_training_episodes':len(ueps),'training_successes':sum(e['success'] for e in ueps),
                    'training_episode_scope':'Continuation256_to1024 only; excludes shared parent',
                    'unfinished_decisions_at1024':u1024['segment_experience']['unfinished_episode_decisions']},
                'reverse_restored':{'evaluations':rpoints,'optimizer_curve':rcurve,'training_by_start':by_start(old_reps+new_reps),
                    'complete_training_episodes':len(old_reps)+len(new_reps),'training_successes':sum(e['success'] for e in old_reps+new_reps),
                    'training_episode_scope':'Retained256_to512 plus recovery02_512_to1024; excludes shared parent and discarded tail',
                    'retired_unfinished_decisions_at512':r512['segment_experience']['unfinished_episode_decisions'],
                    'recovery02_Adam_updates':r1024['optimizer_completed']-r512['optimizer_completed'],
                    'recovery02_new_training_episodes':len(new_reps),'recovery02_new_successes':sum(e['success'] for e in new_reps)}},
        'search':{'return':search['evaluation']['return'],'success':search['evaluation']['success'],
            'evaluation_decisions':search['evaluation']['suffix_decisions'],'globally_optimal':False,
            'history_hash':search['evaluation']['history_hash'],'plan_seal':search['evaluation']['plan_seal']},
        'discarded_original_reverse_work':discarded,
        'recovery':{'restored_decisions':512,'restored_Adam_updates':r512['optimizer_completed'],
            'model_and_optimizer_exactly_restored':True,'new_RNG_and_trajectory':True,
            'memory_only_implementation_change':True,'not_a_seamless_trajectory_continuation':True},
        'training_episode_rows':ueps+old_reps+new_reps,
        'limitations':['Sampled training successes are not independently replayed full-start evaluations.',
            'Optimizer x-axis is retained model decision clock, not total paid work.',
            'Original reverse run failed at RSS cap; its discarded tail and failed recovery01 remain paid separate.',
            'No uncertainty bands are justified by one seed and one task; search is bounded, not proven optimal.',
            'Full cost audit is maintained separately; this renderer creates no new scientific evaluations.'],
        'source_pins':SOURCES,'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def render(r):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'svg.hashsalt':'train045-local-goal-recovery02-final','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(14,9.7),gridspec_kw={'height_ratios':[1,1]})
    fig.subplots_adjust(left=.075,right=.965,top=.86,bottom=.15,wspace=.29,hspace=.39)
    fig.suptitle('Local-goal PPO: practice successes did not become full-start success',x=.075,y=.97,ha='left',fontsize=17,fontweight='bold')
    fig.text(.075,.933,'One TRAIN 045  •  seed 0  •  annotation-assisted SIMULATOR  •  frozen spatial encoder',fontsize=11,color='#45505b')
    ax=axes[0,0];shared=r['shared_parent_evaluations']
    ax.plot([p['training_decisions'] for p in shared],[p['return'] for p in shared],'-o',color=COLORS['shared'],lw=2,label='Shared frozen parent')
    for key,color,marker,label in [('uniform',COLORS['uniform'],'o','Uniform'),('reverse_restored',COLORS['reverse'],'D','Reverse → restored 512')]:
        points=r['arms'][key]['evaluations'][2:]
        ax.plot([p['training_decisions'] for p in points],[p['return'] for p in points],'-' if key=='uniform' else '--',
            marker=marker,ms=6 if key=='uniform' else 9,mfc=color if key=='uniform' else 'none',color=color,lw=1.8,label=label)
    ax.axhline(r['search']['return'],color=COLORS['search'],ls=':',lw=2)
    ax.text(22,r['search']['return']+.3,f"Bounded search: {r['search']['return']:.5f} (goal reached)",color=COLORS['search'],fontsize=10)
    ax.text(480,.8,'512 / 1024: both STOP, return 0',fontsize=9,color='#45505b')
    ax.set(title='A   Evaluated full-start return',xlabel='Retained training decisions',ylabel='Local-task return',xlim=(-25,1080),ylim=(-.95,10.8))
    ax.set_xticks([0,64,256,512,1024]);ax.tick_params(axis='x',labelsize=9)
    ax.grid(axis='y',alpha=.18);ax.legend(loc='center left',bbox_to_anchor=(.01,.47),frameon=False,fontsize=8.5)
    inset=ax.inset_axes([.52,.35,.45,.34]);inset.set_facecolor('#f7f8fa')
    for key,color,mark in [('uniform',COLORS['uniform'],'o'),('reverse_restored',COLORS['reverse'],'D')]:
        p=r['arms'][key]['evaluations'];inset.plot([v['training_decisions'] for v in p],[v['return'] for v in p],
            marker=mark,ms=3,lw=1.2,color=color,ls='-' if key=='uniform' else '--',mfc=color if key=='uniform' else 'none')
    inset.set(xlim=(-20,1050),ylim=(-.58,.08),xticks=[0,512,1024],yticks=[-.5,0]);inset.tick_params(labelsize=7)
    inset.set_title('PPO return detail',fontsize=8);inset.grid(alpha=.15)
    ax=axes[0,1]
    for key,color in [('uniform',COLORS['uniform']),('reverse_restored',COLORS['reverse'])]:
        p=r['arms'][key]['optimizer_curve']
        if key=='uniform':ax.plot([v['training_decisions'] for v in p],[v['Adam_updates'] for v in p],'-o',color=color,ms=3,lw=2,label='Uniform')
        else:
            for segment,ls,label in ((p[:9],'-','Reverse to 512'),(p[8:],'--','Restored 512 → 1024')):
                ax.plot([v['training_decisions'] for v in segment],[v['Adam_updates'] for v in segment],ls,marker='D',color=color,ms=3,lw=2,label=label)
    ax.axvline(512,color='#a4aab1',lw=1,ls=':')
    ax.annotate('Restore 353 Adam steps\nnew RNG + episode',xy=(512,353),xytext=(535,246),fontsize=8.5,color=COLORS['reverse'],arrowprops={'arrowstyle':'-','color':COLORS['reverse']})
    ax.text(1038,415,'415',va='center',fontsize=10,color=COLORS['reverse']);ax.text(1038,338,'338',va='center',fontsize=10,color=COLORS['uniform'])
    ax.set(title='B   Actual Adam updates in retained lineage',xlabel='Retained training decisions',ylabel='Completed Adam updates',xlim=(-20,1135),ylim=(-10,465),xticks=[0,256,512,768,1024])
    ax.grid(alpha=.18);ax.legend(loc='upper left',frameon=False,fontsize=8.5)
    ax=axes[1,0]
    for key,offset,color,label in [('uniform',-.18,COLORS['uniform'],'Uniform'),('reverse_restored',.18,COLORS['reverse'],'Reverse, retained + recovered')]:
        groups=r['arms'][key]['training_by_start']
        rates=[groups[str(s)]['successes']/groups[str(s)]['episodes'] for s in range(12,17)]
        xx=[s+offset for s in range(12,17)];ax.bar(xx,rates,width=.32,color=color,alpha=.85,label=label)
        for x,s,y in zip(xx,range(12,17),rates):
            q=groups[str(s)];ax.text(x,y+.012,f"{q['successes']}/{q['episodes']}",ha='center',va='bottom',fontsize=8.5,color=color)
    ax.set(title='C   Sampled practice: successes / complete episodes',xlabel='Reset start along the qualified path (absolute step)',ylabel='Training episode success rate',xticks=list(range(12,17)),ylim=(0,.46))
    ax.set_xticklabels(['12\nfull start','13','14','15','16\nnearest goal']);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.grid(axis='y',alpha=.18)
    ax.legend(loc='upper left',frameon=False,fontsize=8.5)
    ax=axes[1,1];ax.axis('off');ax.set_title('D   Final full-start comparison',loc='left')
    table=ax.table(cellText=[['Bounded search',f"{r['search']['return']:.5f}",'Yes','5'],['Uniform 1024','0.00000','No','1 (STOP)'],['Reverse 1024, recovered','0.00000','No','1 (STOP)']],
        colLabels=['Method','Return','Goal reached','Decisions'],colWidths=[.44,.2,.2,.17],cellLoc='left',loc='upper left',bbox=[0,.51,1,.40])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('#dce1e6');cell.set_linewidth(.5)
        cell.set_facecolor('#eef2f5' if row==0 else 'white')
        if row==0:cell.set_text_props(weight='bold')
    ax.text(0,.40,'No RL advantage shown on this task.',fontsize=12,weight='bold',color='#38444d')
    ax.text(0,.30,'Full-start endpoints: accepted replay and local reward checks.\nPractice panel: retained continuation 256 → 1024 only;\nshared-parent practice excluded. Uniform 9/368; reverse 13/269.\nRecovery 02 restored the exact 512 policy + Adam state;\nits subsequent RNG and physical trajectory were new.',fontsize=9.5,color='#45505b',va='top',linespacing=1.5)
    fig.text(.075,.057,'Failed original reverse run preserved: at least 320 post-checkpoint decisions and 40 Adam updates were paid,\nthen discarded from checkpoint lineage. They are not spliced into these curves. Six unfinished decisions at 512 were retired.',fontsize=9,color='#45505b',linespacing=1.5)
    fig.text(.075,.018,'Saved scalar evidence only • one task / one seed • no clinical or generalization claim • complete work accounting is reported separately',fontsize=8.5,color='#65717b')
    fig.savefig(HERE/'learning-curves.png',dpi=160,facecolor='white')
    fig.savefig(HERE/'learning-curves.svg',facecolor='white',metadata={'Date':None})
    plt.close(fig)


if __name__=='__main__':
    record=gather();render(record)
    record['render_seconds']=time.monotonic()-STARTED
    record['output_hashes']={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('learning-curves.png','learning-curves.svg')}
    (HERE/'learning-curves.json').write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
    total=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    assert total<=3*1024**2 and time.monotonic()-STARTED<60
    print(json.dumps({'status':'complete','seconds':time.monotonic()-STARTED,'total_bytes_including_cache':total,
        'files':{str(p.relative_to(HERE)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
                 for p in sorted(HERE.iterdir()) if p.is_file()},
        'training_by_start':{k:v['training_by_start'] for k,v in record['arms'].items()}},sort_keys=True))
