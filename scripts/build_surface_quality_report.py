#!/usr/bin/env python3
"""Assemble inspectable software QA receipts and patient screenshots locally."""
import argparse
from html import escape
import json
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();root=args.directory.resolve()
    audit=json.loads((root/'cohort/audit.json').read_text())
    comparison=json.loads((root/'analytic-comparison.json').read_text())
    baseline=json.loads((root/'baseline/rl.json').read_text());current=json.loads((root/'rl-current.json').read_text())
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RessectionLab · surface quality verification</title><style>body{margin:0;background:#09131b;color:#dbe7ed;font:15px/1.55 system-ui}main{max-width:1180px;margin:auto;padding:36px 24px}h1{font-size:30px}h2{margin-top:42px;color:#9cd6ce}h3{margin-top:28px}a{color:#7fcbe7}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #28404b}small{color:#9eb0bc}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img,video{width:100%;border:1px solid #28404b;border-radius:8px}summary{cursor:pointer}.note{background:#142933;padding:16px;border-radius:8px}code{overflow-wrap:anywhere}li{margin:5px 0}</style><main>']
    patients=audit['patients'];sources=all(row['sourceUnchanged'] for row in audit['files'])
    history=sum(len(row.get('savedHistory',[])) for row in audit['files'])
    html.append(f'<h1>Smoother anatomy and scan rendering</h1><p>{len(audit["files"])} local files audited across {len(patients)} independent patient IDs. Original files unchanged: <b>{sources}</b>. Fixed headless RL fingerprints match: <b>{baseline==current}</b>.</p>')
    html.append('<p class="note">These results establish rendering and software behavior. Anatomical accuracy remains unmeasured without suitable reference annotations. Subject 13 has an empty tumor prediction; it remains missing anatomy, and its existing RL admission rejection is retained. Segmentation masks, measurements, source intensities and RL physics are unchanged.</p>')
    html.append('<h2>Analytic surface comparison</h2><p>Mean angular-normal error (1 − cosine) against the analytic sphere or cylinder. Lower is better. Same source masks and physical grids; comparison uses the captured pre-change six-pair smoother.</p><table><tr><th>Fixture</th><th>Raw source</th><th>Previous display</th><th>New display</th><th>Reduction vs previous</th></tr>')
    for row in comparison:html.append(f'<tr><td>{escape(row["kind"])}</td><td>{row["sourceError"]:.5f}</td><td>{row["oldError"]:.5f}</td><td>{row["newError"]:.5f}</td><td>{row["improvementVersusOldPercent"]:.1f}%</td></tr>')
    html.append('</table><h2>Patient views and performance</h2><p>Matched raw-source and smoothed display views use the same camera. Orbit clips are demonstrations. Performance comes from a separate gesture without screenshot encoding; Electron frame subscription still has observer overhead. Values describe the tested machine and views, not a universal FPS floor. Some high-resolution views remain below the 30 FPS target; all measured scan refinements were below two seconds.</p>')
    render_rows=[]
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'report.json').exists():continue
        report=json.loads((folder/'report.json').read_text());name=folder.name;render_rows.append((name,report))
        html.append(f'<h3>{escape(name)}</h3><p>Electron check: <b>{report.get("passed")}</b>. Shader errors: {len(report.get("errors",[]))}.</p>')
        if report.get('failure'):html.append('<pre>'+escape(report['failure'])+'</pre>')
        if (folder/'source-oblique.png').exists() and (folder/'smooth-oblique.png').exists():
            html.append(f'<div class="pair"><figure><a href="{name}/source-oblique.png"><img src="{name}/source-oblique.png" loading="lazy" alt="Unsmoothed source surface"></a><figcaption>Unsmoothed source surface</figcaption></figure><figure><a href="{name}/smooth-oblique.png"><img src="{name}/smooth-oblique.png" loading="lazy" alt="Smoothed anatomy display"></a><figcaption>Smoothed display</figcaption></figure></div>')
        for image in ['volume-anterior.png','ct-volume-anterior.png','rehearsal-reference.png']:
            if (folder/image).exists():html.append(f'<details><summary>{"Rehearsal" if image.startswith("rehearsal") else "CT" if image.startswith("ct-") else "Source scan"} volume view</summary><img src="{name}/{image}" loading="lazy" alt="Acquired scan with anatomy surfaces"></details>')
        if (folder/'orbit.mp4').exists():html.append(f'<details><summary>Orbit demonstration</summary><video controls preload="none" src="{name}/orbit.mp4"></video></details>')
        perf=report.get('orbitWithoutScreenshots',report.get('rotation',{}));fps=perf.get('medianFps');p95=perf.get('p95Ms')
        refinements=[r['milliseconds'] for r in report.get('refinement',[]) if r.get('completed') and r.get('volume')]
        memory=sum(p.get('memory',{}).get('workingSetSize',0) for p in report.get('memory',[]))/1024
        html.append(f'<p>Delivered median FPS: <b>{fps:.1f}</b>; 95th-percentile interval: {p95:.1f} ms. 30 FPS target: {"met" if fps>=30 else "not met in this view"}. ' if fps is not None and p95 is not None else '<p>Frame timing unavailable. ')
        html.append(f'Maximum measured scan refinement: {max(refinements):.0f} ms. ' if refinements else 'Scan refinement unavailable. ')
        html.append(f'Sampled application working set: {memory:.0f} MiB (not peak memory). <a href="{name}/report.json">Raw receipt</a>.</p>')
    html.append(f'<h2>Per-file findings and saved history</h2><p>{history} saved branches replayed where recorded. Files with multiple saves do not count as additional patients. <a href="cohort/audit.json">Full audit and source hashes</a>.</p><table><tr><th>File</th><th>Status / source unchanged</th><th>Source findings</th><th>History</th></tr>')
    for row in audit['files']:
        findings=[]
        for layer in row['layers']:
            q=layer['qa'];text=f'{layer["name"]}: {q["voxelCount"]:,} cells, {q["componentCount"]:,} components'
            if not q['voxelCount']:text+=' — EMPTY / missing anatomy'
            if q['boundaryTouchingVoxels']:text+=f'; {q["boundaryTouchingVoxels"]:,} boundary cells'
            if layer.get('surface'):text+=f'; display displacement ≤{layer["surface"]["maxDisplacementMm"]:.4f} mm; {layer["surface"]["restoredComponents"]} source-restored components'
            findings.append(escape(text))
        error='<br>'+escape(row['error']) if row.get('error') else ''
        html.append(f'<tr><td><code>{escape(row["path"])}</code></td><td>{escape(row["status"])} / {row["sourceUnchanged"]}{error}</td><td>'+ '<br>'.join(findings)+f'</td><td>{len(row.get("savedHistory",[]))} branches</td></tr>')
    html.append('</table><h2>Unavailable sources</h2><p>The following catalog members were not materialized locally and remain untested:</p><ul>')
    for row in audit['unavailable']:html.append('<li>'+escape(row['patient_id'])+'</li>')
    html.append('</ul><h2>Checks and practical limits</h2><ul>')
    for filename,label in [('desktop-final-tests.log','Desktop tests'),('viewer-tests.log','Viewer rerun'),('python-tests.log','Python anatomy, persistence and RL tests')]:
        if (root/filename).exists():
            text=(root/filename).read_text();counts=re.findall(r'(?:ℹ tests |# tests )(\d+)',text);passed=re.findall(r'(\d+) passed',text)
            summary=f'{sum(map(int,counts))} tests' if counts else (passed[-1]+' passed' if passed else 'see log')
            html.append(f'<li><a href="{filename}">{label}</a>: {summary}.</li>')
    html.append('<li>Eleven legacy bridge tests also fail against the captured baseline bridge; the disabled synthetic-case contract was preserved. <a href="baseline/legacy-bridge.log">Baseline failure evidence</a>.</li><li>The historical engineering-fixture save has a pre-existing scene identity/replay incompatibility; its source file remains unchanged. Patient saves passed their history checks.</li><li>Foreground translucent mesh boundaries use depth peeling. Remaining transparent fragments are retained in a sorted tail; ordering inside that tail is approximate. Interaction uses a smaller peel budget; final refinement restores full quality.</li><li>Very large context meshes may retain the original geometry with an explicit fallback. Vessels and tumors retain full tissue-specific smoothing and intersection checks. Initial uncached high-resolution preparation can take tens of seconds or longer; displayed refinement is measured separately.</li><li>No acquired CTA is present in this local cohort. CTA selection gates and transfer-function behavior have software tests; CT and structural-MRI rendering have real-patient captures.</li></ul>')
    html.append('<p><a href="analytic-comparison.json">Analytic measurements</a> · <a href="rl-current.json">Current RL fingerprints</a> · <a href="baseline/rl.json">Baseline RL fingerprints</a></p></main></html>')
    (root/'index.html').write_text(''.join(html))
    print(root/'index.html')


if __name__=='__main__':main()
