#!/usr/bin/env python3
"""One frozen numerical figure; sibling JSON inputs only, no model or curve fitting."""
from pathlib import Path
import hashlib
import json
import resource
import signal
import sys
import time

START=time.monotonic()
HERE=Path(__file__).resolve().parent
PROVENANCE_SHA='82ca1077d871f77d511f5ee40547412a4acdaaca80683dc833a2faf07a85219d'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def timeout(signum,frame): raise TimeoutError('Fixed plotting time allowance exceeded')

def main():
    assert sha(HERE/'provenance.json')==PROVENANCE_SHA, 'Plotting choices changed'
    with (HERE/'started.json').open('x') as stream:
        json.dump({'fixed_provenance_sha256':PROVENANCE_SHA,'attempt':1},stream)
    result={'schema':'hbe-n32-global-fixed-figure-result.v1','status':'failed','provenance_sha256':PROVENANCE_SHA,'attempts':1}
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,115)
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        provenance=json.loads((HERE/'provenance.json').read_text())
        for name,binding in provenance['inputs'].items():
            assert sha(HERE/name)==binding['sha256'], f'Changed compact input {name}'
        review=json.loads((HERE/'verification.json').read_text())
        comparison=json.loads((HERE/'comparison.json').read_text())
        assert review['comparison_sha256']==sha(HERE/'comparison.json')
        assert review['status']=='saved_execution_verified_negative_spatial_diagnostic'
        assert review['complete_comparison_reproduced'] is True
        assert comparison['spatial_convergence_accepted'] is False and comparison['calibration_released'] is False
        assert comparison['measured_data_accessed'] is False and comparison['physical_validation_pass'] is None
        states=comparison['states'];assert [s['state'] for s in states]==list(range(61))
        assert states[0]['status']=='rest/not_estimated'
        x=np.arange(61)
        triplets=('N8-N12-N16','N12-N16-N24','N16-N24-N32')
        values=lambda key:np.array([np.nan if (v:=s.get(key)) is None else float(v)*1000 for s in states])
        envelope=values('two_latest_limit_envelope_N')
        orders={key:np.array([np.nan if (v:=s['orders'].get(key,{}).get('order')) is None else float(v) for s in states]) for key in triplets}
        single=np.array([np.nan if (v:=s['orders'].get(triplets[-1],{}).get('absolute_remaining_indicator_N')) is None else float(v)*1000 for s in states])
        allowance=float(comparison['force_allowance_N'])*1000
        assert np.isnan(single[0]) and np.isnan(envelope[0])
        assert np.isfinite(single[1:]).all() and np.isfinite(envelope[1:]).all()
        assert [int(i) for i in np.flatnonzero(np.isnan(orders[triplets[0]]))]==list(range(36))
        assert all(np.isfinite(orders[k][1:]).all() for k in triplets[1:])
        exceedances=[int(i) for i in np.flatnonzero(envelope>allowance)]
        assert exceedances==comparison['remaining_exceedance_states']==list(range(49,61))
        assert review['conclusion']['force_allowance_N']==comparison['force_allowance_N']
        assert review['conclusion']['endpoint_two_latest_limit_envelope_N']==comparison['endpoint']['two_latest_limit_envelope_N']
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,
                             'figure.facecolor':'white','savefig.facecolor':'white'})
        fig=plt.figure(figsize=tuple(provenance['fixed_figure']['size_inches']))
        ink,muted,teal,orange,purple='#172839','#536575','#007e87','#c96523','#8370a3'
        fig.text(.065,.948,'Numerical model study · N32 spatial diagnostic',fontsize=21,weight='bold',color=ink)
        fig.text(.065,.906,'61 saved load states · no measured-force validation or calibration',fontsize=12,color=muted)
        ax=fig.add_axes([.080,.515,.600,.280])
        order_ax=fig.add_axes([.080,.180,.600,.215],sharex=ax)
        for axis in (ax,order_ax):
            axis.set_xlim(0,60);axis.set_xticks(np.arange(0,61,10));axis.set_axisbelow(True)
            axis.grid(color='#e4e9ed',linewidth=.6)
            axis.spines[['right','top']].set_visible(False)
            axis.spines[['left','bottom']].set_color('#c2cbd3')
            axis.tick_params(labelsize=9,colors=muted)
        ax.set_ylim(0,1.08);ax.set_yticks(np.arange(0,1.01,.2))
        ax.set_ylabel('Remaining-force diagnostic (mN)',color=ink)
        ax.set_title('a  Latest single-model estimate and retained two-limit envelope',loc='left',fontsize=11,weight='bold',color=ink,pad=38)
        ax.axvspan(48.5,60,color='#f9e9dc',zorder=0)
        ax.axhline(allowance,color=ink,lw=1.2,ls=(0,(4,3)),label='Original fixed allowance')
        ax.plot(x,single,color=teal,lw=1.8,marker='o',ms=2.8,label='Latest model: N16/24/32')
        ax.plot(x,envelope,color=orange,lw=1.8,marker='o',ms=2.8,label='Two-latest-limit envelope')
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.025),ncols=2,fontsize=8.8,frameon=False,
                  borderaxespad=0,handlelength=2,columnspacing=1.2)
        ax.text(6,.97,'Envelope exceeds allowance at states 49–60',color=orange,fontsize=9,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2})
        ax.set_xlabel('Saved load state',color=ink,labelpad=7)

        order_ax.set_ylim(0,1.0);order_ax.set_yticks([0,.25,.5,.75,1.0])
        order_ax.set_ylabel('Observed power-law order',color=ink)
        order_ax.set_xlabel('Saved load state (state 0: rest, not estimated)',color=ink,labelpad=7)
        order_ax.set_title('b  Triplet-specific orders; unavailable estimates remain missing',loc='left',fontsize=11,weight='bold',color=ink,pad=28)
        order_ax.axvspan(.5,35.5,color='#f0f1f5',zorder=0)
        for key,color,marker in zip(triplets,(purple,'#5d738c',teal),('s','^','o')):
            order_ax.plot(x,orders[key],color=color,lw=1.5,marker=marker,ms=2.8,label=key.replace('-',' / '))
        order_ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),ncols=3,fontsize=8.8,frameon=False,
                        borderaxespad=0,handlelength=2,columnspacing=1.2)
        order_ax.text(17.5,.15,'Oldest triplet: 35 ineligible states\n(no admissible positive root)',ha='center',va='center',
                      fontsize=9,color='#676176',bbox={'facecolor':'#f0f1f5','edgecolor':'none','pad':2})
        # A fixed side column keeps all endpoint annotations away from plotted samples.
        side=fig.add_axes([.735,.190,.23,.635]);side.set_axis_off()
        side.text(0,.98,'ENDPOINT · STATE 60',fontsize=10,weight='bold',color=muted)
        side.text(0,.88,f'{envelope[-1]:.5f} mN',fontsize=22,weight='bold',color=orange)
        side.text(0,.835,'Retained two-latest-limit envelope',fontsize=9,color=muted)
        side.text(0,.730,f'{allowance:.5f} mN',fontsize=19,weight='bold',color=ink)
        side.text(0,.685,'Original fixed allowance',fontsize=9,color=muted)
        side.text(0,.580,f'{single[-1]:.5f} mN',fontsize=19,weight='bold',color=teal)
        side.text(0,.535,'Latest single-model estimate',fontsize=9,color=muted)
        side.axhline(.46,color='#d9dfe5',linewidth=.8)
        side.text(0,.40,'Spatial criterion remains unmet',fontsize=10,weight='bold',color=orange)
        side.text(0,.325,'12 / 60 nonrest states exceed\nthe retained-envelope allowance.',fontsize=10,color=ink,linespacing=1.5,va='top')
        side.text(0,.235,'Eligible nonrest states',fontsize=9.5,weight='bold',color=muted)
        side.text(0,.115,'N8 / 12 / 16       25 / 60\nN12 / 16 / 24     60 / 60\nN16 / 24 / 32     60 / 60',fontsize=9.5,color=ink,linespacing=1.7)
        side.text(0,.015,'State 0 is retained as not estimated.',fontsize=8.4,color=muted)
        fig.text(.080,.092,'Conditional power-law model sensitivity; neither curve is a rigorous continuum error bound. No finest-level temporal-convergence claim.',fontsize=9,color=ink)
        fig.text(.080,.065,'Saved numerical mechanics comparison only. No primitive logs, biological force curves, patient data, fitting or solver execution used for this figure.',fontsize=9,color=ink)
        fig.text(.080,.035,'Comparison SHA256: 0c989b6de65b14df…  ·  Independent saved-result review: 10a9c97d61f95440…  ·  Null estimates are not plotted as zero.',fontsize=8,color=muted)
        png,pdf=HERE/'n32-spatial-diagnostic.png',HERE/'n32-spatial-diagnostic.pdf'
        fig.savefig(png,dpi=provenance['fixed_figure']['png_dpi'],metadata={'Software':'RessectionLab fixed numerical diagnostic'})
        fig.savefig(pdf,metadata={'Title':'Numerical model study: N32 spatial diagnostic','Creator':'RessectionLab plot.py','CreationDate':None,'ModDate':None})
        plt.close(fig)
        for name,binding in provenance['inputs'].items(): assert sha(HERE/name)==binding['sha256']
        result.update(status='rendered',wall_seconds=time.monotonic()-START,inputs_unchanged=True,
            plotted_state_count=61,remaining_exceedance_states=exceedances,
            missing_order_states={k:[int(i) for i in np.flatnonzero(np.isnan(v))] for k,v in orders.items()},
            os_reported_worker_maxrss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
            source_sha256=sha(Path(__file__)),versions={'numpy':np.__version__,'matplotlib':matplotlib.__version__},
            outputs={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in (png,pdf)})
    except Exception as exc:
        result.update(failure_type=type(exc).__name__,failure=str(exc),wall_seconds=time.monotonic()-START)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        (HERE/'render-result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['status']=='rendered' else 1

if __name__=='__main__':raise SystemExit(main())
