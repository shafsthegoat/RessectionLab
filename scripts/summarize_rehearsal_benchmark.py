#!/usr/bin/env python3
"""Summarize retained raw foreground reports without changing measurements."""
import argparse
import json
from pathlib import Path
import statistics


def percentile(values, fraction=.95):
    if not values:
        return None
    values=sorted(values)
    position=(len(values)-1)*fraction
    low=int(position)
    return values[low]+(values[min(low+1,len(values)-1)]-values[low])*(position-low)


def describe(values):
    return {'n':len(values),'median':statistics.median(values) if values else None,'p95':percentile(values)}


def summarize(report):
    runs=report['runs']
    seeks=[r for r in runs if r['name'].startswith('seek-')]
    cameras=[r for r in runs if r['name'].startswith('warm-camera-')]
    poses=[r for r in runs if r['name'].startswith('tool-pose-')]
    gaps=[]
    for run in cameras:
        frames=[f for f in run['frames'] if f.get('interactive')]
        gaps.extend(b['at']-a['at'] for a,b in zip(frames,frames[1:]))
    all_frames=[f for r in runs for f in r.get('frames',[])]
    warm_frames=[f for r in cameras+poses+seeks for f in r.get('frames',[])]
    query_times=[sample['milliseconds'] for r in cameras for sample in r.get('gpuTimes',[]) if sample.get('completed') and 'milliseconds' in sample]
    pick_times=[sample['milliseconds'] for r in runs if r['name']=='picking' for sample in r.get('picks',[])]
    pose_next=[]
    pose_work=[]
    for run in poses:
        for event in run.get('inputs',[]):
            frame=next((f for f in run['frames'] if f['at']>=event['at']),None)
            if frame:pose_next.append(frame['at']-event['at'])
        work=[f.get('displayWork') for f in run['frames'] if f.get('displayWork')]
        if work:pose_work.append({key:work[-1].get(key,0)-work[0].get(key,0) for key in work[-1]})
    rss=[]
    for sample in report.get('memory',[]):
        rss.append(sum(p['memory'].get('workingSetSize',0)*1024 for p in sample['processes'])+(sample.get('backend') or {}).get('rssKiB',0)*1024)
    primary_memory_indices=[0]+[i+1 for i,run in enumerate(runs) if not run['subscribed'] and not run['name'].startswith('verify-') and run['name']!='reopen']
    primary_runs=[run for run in runs if not run['subscribed'] and not run['name'].startswith('verify-') and run['name']!='reopen']
    tissue_frames=[f for r in runs if r['name']=='tissue-next-local-action' for f in r.get('frames',[])]
    result={
        'label':report['label'],'passed':report.get('passed'),
        'source':report['source'],'scene':report.get('scene'),'display':report.get('display'),'implementationIdentity':report.get('implementationIdentity'),
        'startupMs':report.get('startupMs'),'enterRehearsalMs':next((r['consistentDisplayAt']-r['inputAt'] for r in runs if r['name']=='enter-rehearsal'),None),
        'seekConsistentDisplayMs':describe([r['consistentDisplayAt']-r['inputAt'] for r in seeks]),
        'seekHandlerMs':describe([i['ms'] for r in seeks for i in r.get('ipc',[]) if i.get('command')=='seek']),
        'inputToAppliedStateMs':describe([r['inputToAppliedStateMs'] for r in seeks if r.get('inputToAppliedStateMs') is not None]),
        'warmInteractiveSubmissionGapMs':describe(gaps),
        'warmCpuSubmissionMs':describe([f['cpuMs'] for f in warm_frames]),
        'gpuElapsedMs':describe(query_times),'pickCpuMs':describe(pick_times),
        'poseInputToNextSubmissionMs':describe(pose_next),'poseResourceWorkDeltas':pose_work,
        'warmLongTasks':sum(len(r.get('longTasks',[])) for r in cameras+poses+seeks),
        'largestSampledHeapBytes':max((r.get('heap',0) or 0 for r in runs),default=0),
        'largestSampledProcessRssBytes':max(rss,default=None),
        'commonPrimaryLargestSampledRssBytes':max((rss[i] for i in primary_memory_indices if i<len(rss)),default=None),
        'commonPrimaryLargestSampledHeapBytes':max((r.get('heap',0) or 0 for r in primary_runs),default=0),
        'seekRunStartToHandlerStartMs':describe([i['started']-r['started'] for r in seeks for i in r.get('ipc',[]) if i.get('command')=='seek']),
        'perProcessPeakWorkingSetBytes':{kind:max((p['memory'].get('peakWorkingSetSize',0)*1024 for sample in report.get('memory',[]) for p in sample['processes'] if p['type']==kind),default=0) for kind in ['Browser','GPU','Tab','Utility']},
        'largestKnownRendererAllocationBytes':max((f.get('estimatedAllocation',{}).get('totalKnownBytes',0) for f in all_frames),default=0) or None,
        'sameCaseSeekReadAssetBytes':sum(i.get('bytes',0) for r in seeks for i in r.get('ipc',[]) if i['channel']=='research:readAsset'),
        'seekJsonResponseBytes':sum(i.get('responseJsonBytes',0) for r in seeks for i in r.get('ipc',[])),
        'completeActionConsistentMs':describe([r['consistentDisplayAt']-r['inputAt'] for r in runs if r['name']=='complete-inspect-action']),
        'tissueApplyConsistentMs':describe([r['consistentDisplayAt']-r['inputAt'] for r in runs if r['name']=='tissue-apply']),
        'tissueRevisitConsistentMs':describe([r['consistentDisplayAt']-r['inputAt'] for r in runs if r['name'].startswith(('tissue-undo-','tissue-redo-'))]),
        'tissuePreparationWork':[w for r in runs if r['name'].startswith('tissue-') for w in r.get('work',[]) if w.get('kind')=='modeled-surface-preparation'],
        'lastLabelUploads':next((f['labelUploads'] for f in reversed(all_frames) if f.get('labelUploads')),None),
        'tissueLabelUploads':next((f['labelUploads'] for f in reversed(tissue_frames) if f.get('labelUploads')),None),
        'tissueSurfaceCache':next((f['surfaceCache'] for f in reversed(tissue_frames) if f.get('surfaceCache')),None),
        'lastSurfaceCache':next((f['surfaceCache'] for f in reversed(all_frames) if f.get('surfaceCache')),None),
        'lastPicking':next((f['picking'] for f in reversed(all_frames) if f.get('picking')),None),
        'primaryTimingForeground':report.get('primaryTimingForeground'),
        'primaryTimingVisible':report.get('primaryTimingVisible'),'monitor':report.get('monitor'),
        'exportBytes':report.get('exportBytes'),'checks':report.get('checks'),
        'quality':{'adaptiveFrames':sum(bool(f.get('surfaceQuality',{}).get('adaptive')) for f in all_frames if f.get('surfaceQuality')),
                   'settledDirectFrames':sum(f.get('surfaceQuality') is not None and not f['surfaceQuality']['adaptive'] for f in all_frames)},
    }
    result['acceptance']={'warmSeekP95Under100ms':(result['seekConsistentDisplayMs']['p95'] or float('inf'))<100,
                          'interactiveGapP95Under33ms':(result['warmInteractiveSubmissionGapMs']['p95'] or float('inf'))<33}
    return result


def aggregate(reports):
    """Pool repeated runs only within the same source/history/workload/arm."""
    groups={}
    for report in reports:
        phase=report['label'].rsplit('-',1)[-1]
        tissue=any(run['name']=='tissue-apply' for run in report['runs'])
        display_key=json.dumps([report.get('window'),report.get('display'),report.get('monitor',{}).get('selected',{}).get('id'),report.get('enabledLayers')],sort_keys=True)
        key=(report['source']['sha256'],report.get('historySteps'),tissue,phase,display_key)
        groups.setdefault(key,[]).append(report)
    result=[]
    for (source,history,tissue,phase,display_key),items in groups.items():
        runs=[run for item in items for run in item['runs']]
        duration=lambda names:[r['consistentDisplayAt']-r['inputAt'] for r in runs if any(r['name'].startswith(name) for name in names)]
        gaps=[]
        for run in runs:
            if not run['name'].startswith('warm-camera-'):continue
            frames=[f for f in run['frames'] if f.get('interactive')]
            gaps.extend(b['at']-a['at'] for a,b in zip(frames,frames[1:]))
        summaries=[summarize(item) for item in items]
        result.append({'sourceSha256':source,'caseId':items[0].get('scene',{}).get('sceneId'),
            'historySteps':history,'tissue':tissue,'phase':phase,'runs':[i['label'] for i in items],
            'displayAndLayers':json.loads(display_key),
            'implementationIdentities':[dict(identity) for identity in {tuple(sorted(i.get('implementationIdentity',{}).items())) for i in items}],
            'allPassed':all(i.get('passed') is True for i in items),
            'allForeground':all(i.get('primaryTimingForeground') is True for i in items),
            'allVisible':all(i.get('primaryTimingVisible') is True or i.get('primaryTimingForeground') is True for i in items),
            'harnessSha256':sorted({i.get('harnessSha256','unavailable') for i in items}),
            'startupFirstCanvasMs':describe([i['startupMs'] for i in items]),
            'enterRehearsalConsistentMs':describe(duration(('enter-rehearsal',))),
            'warmSeekConsistentMs':describe(duration(('seek-',))),
            'interactiveSubmissionGapMs':describe(gaps),
            'completeInspectActionMs':describe(duration(('complete-inspect-action',))),
            'tissuePreviewMs':describe(duration(('tissue-preview',))),
            'tissueApplyMs':describe(duration(('tissue-apply',))),
            'tissueNextLocalActionMs':describe(duration(('tissue-next-local-action',))),
            'tissueRevisitMs':describe(duration(('tissue-undo-','tissue-redo-'))),
            'pickCpuMs':describe([p['milliseconds'] for r in runs if r['name']=='picking' for p in r.get('picks',[])]),
            'largestSampledAggregateRssBytes':max((s['largestSampledProcessRssBytes'] or 0 for s in summaries),default=0),
            'commonPrimaryLargestSampledRssBytes':max((s['commonPrimaryLargestSampledRssBytes'] or 0 for s in summaries),default=0),
            'largestSampledHeapBytes':max((s['largestSampledHeapBytes'] or 0 for s in summaries),default=0),
            'commonPrimaryLargestSampledHeapBytes':max((s['commonPrimaryLargestSampledHeapBytes'] or 0 for s in summaries),default=0),
            'exportBytes':[i.get('exportBytes') for i in items],
            'tissueFinalStateIds':sorted({i['tissueFinal']['stateId'] for i in items if 'tissueFinal' in i})})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--include-rejected',action='store_true',help='Explicitly summarize failed/incomplete diagnostic reports; never use them as accepted comparison samples.')
    args=parser.parse_args()
    reports=[json.loads(path.read_text()) for path in args.reports]
    rejected=[report.get('label','unknown') for report in reports if report.get('passed') is not True or not(report.get('primaryTimingForeground') is True or report.get('primaryTimingVisible') is True)]
    if rejected and not args.include_rejected:
        parser.error('Rejected or incomplete reports require --include-rejected: '+', '.join(rejected))
    rows=[summarize(report) for report in reports]
    args.output.write_text(json.dumps({'definitions':{
        'consistent':'Observed pending/mesh completion plus next animation callbacks; CPU submission, not actual screen presentation.',
        'memory':'Largest sampled RSS/heap, not peak. Known renderer allocation excludes driver and default framebuffer.',
        'gpu':'Only asynchronous non-disjoint timer queries; empty baseline samples mean unavailable.',
        'pose':'Input-to-next-renderer-submission proxy; source data and authoritative simulation are separate.'},'reports':rows,'aggregates':aggregate(reports)},indent=2)+'\n')
    print('| Run | Seek median / p95 ms | Interactive gap p95 ms | Action ms | Sampled RSS MiB |')
    print('| --- | --- | --- | --- | --- |')
    for row in rows:
        seek=row['seekConsistentDisplayMs'];gap=row['warmInteractiveSubmissionGapMs']['p95'];action=row['completeActionConsistentMs']['median'];rss=row['largestSampledProcessRssBytes']
        fmt=lambda value:'n/a' if value is None else f'{value:.1f}'
        print(f"| {row['label']} | {fmt(seek['median'])} / {fmt(seek['p95'])} | {fmt(gap)} | {fmt(action)} | {fmt(rss/1024**2 if rss else None)} |")


if __name__=='__main__':main()
