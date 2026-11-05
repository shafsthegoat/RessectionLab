#!/usr/bin/env python3
"""Build the local integrated-suite handoff from exact published evidence."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file, volume_info


def build(root=None):
    root=checked_root(root,required_bytes=100*1024**2)
    def read(relative):return json.loads(safe_path(root,relative).read_text())
    def binding(relative):
        path=safe_path(root,relative)
        return {'path':str(path.relative_to(root)),'sha256':sha256_file(path)}
    current=read('catalog/release.json');base_path='catalog/releases/2026-10-31-data-foundation-v1.json';base=read(base_path)
    authority=read('integrated-v1/qc/independent/release-decisions-v1.json')
    if current.get('inherited_release',{}).get('sha256')!=sha256_file(root/base_path):raise ValueError('A reviewed append-only release is required')
    old={row['case_id']:row for row in base['cases']};now={row['case_id']:row for row in current['cases']}
    if any(now.get(key)!=row for key,row in old.items()):raise ValueError('Prior accepted rows changed')
    additions=[row for row in current['cases'] if row['case_id'] not in old]
    if len(additions)!=7:raise ValueError('This handoff expects the reviewed five CTA and two healthy cases, without counting replay copies')
    expected={'TopCoW:'+n for n in ['004','006','012','014','018']}|{'IXI:077','IXI:081'}
    if {row['person_id'] for row in additions}!=expected or not expected<={row['person_id'] for row in old.values()}:
        raise ValueError('Integrated suite must reuse the seven declared existing source families')
    ui_summary=read('integrated-v1/app/final-ui-summary-v1.json')
    ui={row['subject']:row for row in ui_summary['cases'] if row['passed']}
    replay_ui={row['subject']:row for row in read('integrated-v1/app/replay-ui-summary-v1.json')['cases'] if row['passed']}
    vascular_path='integrated-v1/probes/006-SSS-corridor-execution-v2/report.json'
    vascular=read(vascular_path);vascular_receipt=vascular['source_specific_vascular_corridor_receipt']
    if (vascular_receipt['withinLimit'] or vascular_receipt['maxNearMm3']!=0 or vascular_receipt['marginMm']!=0
            or not all(vascular[k] for k in ['preview_preserved_state','source_case_preserved','source_labels_preserved','original_files_preserved'])
            or vascular['source_masks_modified'] or vascular['target_reached']):
        raise ValueError('Source SSS negative evidence differs from the reviewed task')
    people=[]
    for row in additions:
        if sha256_file(safe_path(root,row['path']))!=row['sha256'] or sha256_file(safe_path(root,row['review_path']))!=row['review_sha256']:
            raise ValueError('Published package/review changed')
        receipt=read(row['review_path'])
        healthy=row['person_id'].startswith('IXI:');subject=row['person_id'].split(':')[-1]
        app=read(ui[subject]['path']+'/report.json')
        if not app['passed'] or app['windowEverVisible'] or app['inputSha256']!=row['sha256']:raise ValueError('Missing exact-byte hidden app proof')
        prep_dir='integrated-v1/cases/'+row['native_case_id'];prep=read(prep_dir+'/report.json')
        person={'person_id':row['person_id'],'case_id':row['case_id'],'path':row['path'],'sha256':row['sha256'],
            'source':row['source'],'source_version':receipt.get('source_version'),'license':receipt.get('license'),
            'role':row['role'],'review':binding(row['review_path']),'capabilities':row['capabilities'],'limitations':row['limitations'],
            'before':'Separate native MRI/MRA source inspection' if healthy else 'CTA and supplied partial vascular annotations; source inspection',
            'after':'Reviewed rigid derived MRA/vessels with synthetic geometry target; approximate overlay inspection' if healthy else 'Reviewed partial brain/CT-skull/vessels/target; conditional bounded negative drilling/access workflow',
            'native_bytes':row['bytes'],'preparation':binding(prep_dir+'/report.json'),
            'app':binding(ui[subject]['path']+'/report.json'),'app_renderer_ready_ms':app['coldReadyMs'],
            'app_total_seconds':ui[subject]['seconds'],'measured_skull_CT':not healthy,'clinical_validation':False}
        if healthy:
            reg=read(f'integrated-v1/healthy/IXI{subject}/rigid-v1/registration.json')
            person.update(registration_seconds=reg['seconds'],registration=binding(f'integrated-v1/healthy/IXI{subject}/rigid-v1/registration.json'),
                source_receipt=binding(f'integrated-v1/healthy/IXI{subject}/source/receipt.json'),
                screenshot=ui[subject]['path']+'/01b-reviewed-registered-fusion.png',target_reach_supported=False,
                withheld=['precise navigation','clinical vascular clearance','measured skull drilling','full rehearsal'])
        else:
            rel=f'integrated-v1/rebinds/{subject}-guarded-full-prefix-v1/report.json';reb=read(rel)
            if reb['source_sha256']!=row['sha256'] or not all(reb[k] for k in ['all_physical_outcomes_exact','all_physical_checkpoint_fields_exact','cold_public_replay_exact']):
                raise ValueError('Guarded physical history does not match accepted case')
            if reb['new_actions_added'] or reb['ignored_physical_state_fields']:raise ValueError('Rebind altered the proven physical sequence')
            anatomy=read(f'integrated-v1/anatomy/TopCoW-{subject}/enclosure-v1/provenance.json')
            replay_app=read(replay_ui[subject]['path']+'/report.json')
            if (not replay_app['passed'] or replay_app['windowEverVisible'] or replay_app['inputSha256']!=reb['saved_sha256']
                    or replay_app.get('physicalReplay',{}).get('stateId')!=reb['final_state_id']
                    or replay_app['physicalReplay']['position']!=reb['full_prefix_action_count']):
                raise ValueError('Actual desktop did not verify the exact physical prefix')
            metrics=reb['metrics'];removed=metrics['removedByLabelMm3']
            person.update(physical_replay=binding(rel),replay_path=reb['saved'],replay_sha256=reb['saved_sha256'],
                source_receipt=binding(f'catalog/topbrain/{subject}-selective-extraction.json'),
                physical_replay_app=binding(replay_ui[subject]['path']+'/report.json'),
                skull_removed_mm3=removed.get('estimated_skull',0),brain_removed_mm3=removed.get('brain',0),target_removed_mm3=metrics['targetRemovedMm3'],
                target_reach_supported=False,physical_clock_seconds=metrics['elapsedSeconds'],
                historical_modeled_temperature_max_c=reb['trajectory_max_per_action_recorded_temperature_C'],
                final_spatial_modeled_temperature_max_c=metrics['drillMaxTemperatureC'],
                anatomy_seconds=anatomy['elapsed_seconds'],native_prepare_seconds=prep['prepare_seconds'],
                native_load_and_scene_seconds=prep['production_load_and_scene_seconds'],
                screenshot=replay_ui[subject]['path']+'/02-rehearsal.png',
                withheld=['intact-head access','simulated scalp/dura opening','target reach','clinical safety','full brain/skull/vascular coverage'])
        people.append(person)
    document={'schema':'ressectionlab-integrated-suite-handoff-v1','generated_utc':datetime.now(timezone.utc).isoformat(),
        'release':binding('catalog/releases/'+current['release_id']+'.json'),'release_id':current['release_id'],
        'prior_release':binding(base_path),'prior_rows_preserved':True,
        'full_goal_complete':authority.get('full_goal_complete',False),
        'counts':{'packages_before':base['case_count'],'packages_after':current['case_count'],'source_person_families':current['person_family_count'],
            'synthetic_variants_before':base['synthetic_variant_count'],'synthetic_variants_after':current['synthetic_variant_count'],
            'new_source_people':0,'integrated_CTA_people':5,'healthy_approximate_overlay_people':2,'acquired_case_target_reach_successes':0,
            'exact_acquired_tumor_MRI_skull_CT_dedicated_angiography_subjects':0},
        'people':people,'independent_review':binding('integrated-v1/qc/independent/release-decisions-v1.json'),
        'vascular_negative_companion':{'report':binding(vascular_path),
            'native':binding(Path(vascular['saved']).relative_to(root)),
            'production_endpoint':vascular['production_endpoint'],'source_SSS_near_cells':len(vascular_receipt['nearSourceCells']),
            'source_SSS_near_mm3':vascular_receipt['nearMm3'],'within_frozen_zero_crossing_limit':False,
            'native_preview_reason':vascular['native_public_preview']['reasonCode'],
            'scope':'Detached source-cell geometric corridor restriction; native preview is additionally obstructed by brain/skull. No reached SSS or uniquely vascular native first refusal.',
            'new_target_variants':0,'new_people':0},
        'hardware':read('catalog/reporting/summary-final.json')['hardware'],'storage':volume_info(root),
        'source_extension':{'prior_qualification':'catalog/source-research.json',
            'clinical_CTA':'Existing TopBrain-v3-2026-08-17 / TopCoW TRAIN clinical source; source-specific noncommercial/attribution terms retained; not healthy',
            'healthy_images':'Existing IXI TRAIN images under CC BY-SA 3.0; acquired T1/MRA timing remains unknown',
            'new_vascular_reference':{'official_metadata':binding('integrated-v1/healthy/source/zenodo-17393202.json'),
                'record_id':17393202,'version':'1.0','license':'CC BY 4.0','evidence':'model-assisted/manually refined source reference; not clinical vessel truth',
                'new_source_people':0,'selected_annotation_people':['IXI:077','IXI:081']}},
        'further_cleanup_this_cycle':False,'prior_cleanup_report':'catalog/reporting/report-final.md',
        'clinical_validation':False,'rl_training':False,'uploads':False}
    destination=root/'integrated-v1';atomic_json(destination/'handoff.json',document)
    lines=['# Integrated acquired-anatomy suite', '',
        f"Accepted release **{current['release_id']}**: {current['case_count']} native packages, {current['person_family_count']} source person families, {current['synthetic_variant_count']} synthetic variants. The previous 355 rows and independent receipts are preserved exactly. These seven additions reuse existing people.", '',
        '**Five CTA people now support conditional local skull drilling and negative access regression. None of the finite acquired-case sequences reaches its synthetic target.** Two healthy people support reviewed MRA/vessel overlay inspection with unresolved alignment accuracy; they have no skull CT. This is an accepted engineering milestone; the full target-reach rehearsal goal remains incomplete. Review is AI engineering/data review, not clinical validation.', '',
        '| Person | Previous capability → new supported task | Removed estimated skull / brain envelope / target (mm³) | Ready-to-open package |',
        '|---|---|---:|---|']
    for p in people:
        removal='— (overlay only)' if not p['measured_skull_CT'] else f"{p['skull_removed_mm3']:.2f} / {p['brain_removed_mm3']:.3f} / {p['target_removed_mm3']:.0f}"
        lines.append(f"| {p['person_id']} | {p['before']} → {p['after']} | {removal} | [{p['case_id']}](../{p['path']}) |")
    lines += ['', '## Physical and evidence limits', '',
        'CTA source intensities, affine, original files and supplied vessel arrays are unchanged. The partial superior brain envelope includes CSF uncertainty; its inferior cutoff and source crop faces remain unknown. The dense CT skull estimate is locally reviewed, not whole-skull ground truth. Targets are geometry only, with no invented CT appearance or mass effect. The source clinical cohort is not healthy.', '',
        'The conditional stage assumes prior scalp exposure and unmodeled dural opening. The UI displays that acquired CTA is a baseline reference. The full nominal tools remain fixed through approach, operation, parking and withdrawal; production task-box guards reject unknown-space actions without mutation. Autonomous planning/preparation is not admitted for these guarded tasks.', '',
        'Actual drill/CUSA sequences remove retained bone and stop at mixed skull/brain material boundaries. The finite 006 offset continuation retains the complete prefix and also stops. This is evidence about those exact protocols, not proof that every approach is impossible. Historical uncalibrated modeled temperatures reach about 124.3 °C; runs are not thermally safe claims. Final spatial maxima are reported separately in handoff.json.', '',
        f"Primary corridors contain no supplied vascular labels, so they alone do not demonstrate vascular causality. The independently reviewed 006 sinus companion uses the existing production EncounterTracker source-cell endpoint: **{len(vascular_receipt['nearSourceCells'])} retained supplied SSS cells / {vascular_receipt['nearMm3']:.6f} mm³** violates a prospectively frozen 0 mm-margin / 0 mm³-crossing constraint. The separate public placement preview returns {vascular['native_public_preview']['reasonCode']}; brain/skull contacts are also retained. This is a detached geometric corridor restriction, not a reached SSS maneuver or uniquely vascular native first refusal. Missing vascular labels never certify free space. [Exact query and source bindings](../{vascular_path}). The metadata-only companion adds zero target variants or people and is excluded from accepted package counts.", '',
        'Healthy rigid registration preserves the T1 grid, resamples images linearly and labels/coverage with nearest neighbor using the same physical transform. The app uses identity mapping for the already resampled derived MRA; original native MRA stays separate. Quantitative patient alignment error, precise navigation, clinical clearance and skull/drilling claims remain unsupported.', '',
        '## Costs on this machine', '',
        'Apple M2 Pro, 10 CPU threads, 16 GiB RAM. Renderer-ready timing starts after the hidden window loads, and includes source/render settling; it is not total process startup. These are one-run measurements under concurrent local work, not benchmark distributions.', '',
        '| Person | Native MB | Anatomy / registration s | Native build s | Renderer-ready s |', '|---|---:|---:|---:|---:|']
    for p in people:
        prep=p.get('native_prepare_seconds');lines.append(f"| {p['person_id']} | {p['native_bytes']/1e6:.1f} | {p.get('anatomy_seconds',p.get('registration_seconds',0)):.2f} | {prep:.2f} | {p['app_renderer_ready_ms']/1000:.2f} |" if prep is not None else f"| {p['person_id']} | {p['native_bytes']/1e6:.1f} | {p['registration_seconds']:.2f} | not separately timed | {p['app_renderer_ready_ms']/1000:.2f} |")
    lines += ['', '## Open, replay and inspect', '',
        'Set `RESSECTIONLAB_DATA_ROOT` to the mounted canonical data directory and run `./scripts/launch_poc.sh` from the primary checkout. Use **Cases** for canonical accepted additions. **Data → Blend registered modality** exposes only the reviewed derived MRA. **Rehearse** displays conditional scope and task restrictions. Use **Open → Verify saved replay** for the physical histories below; replay copies are evidence, not additional variants.', '']
    for p in people:
        if 'replay_path' in p:lines.append(f"- {p['person_id']}: [physical replay](../{p['replay_path']}); [exact history/guard/clock proof](../{p['physical_replay']['path']})")
    lines += ['', 'Every person has actual source/native/mask review, hidden Electron rendering, save/reopen and relevant fusion/rehearsal evidence. The independent decisions and checksummed case sidecars state the exact allowed capabilities. Earlier wrong UI provenance captions are retained as superseded bug evidence, not the final acceptance captures. Historical native preparation flags remain pending; the published independent receipt is authoritative.', '',
        '## Representative production application captures', '']
    for person in ['TopCoW:006','IXI:077']:
        p=next(p for p in people if p['person_id']==person);lines += [f"### {person}",'',f"![Actual hidden application capture](<{root/p['screenshot']}>)",'']
    lines += ['## Reproduction and retained failures', '',
        'Repository instructions: `docs/data-integrated-rehearsal.md`; [exact per-person reproduction commands](reproduce-commands.json). Source/anatomy/target/domain reports bind original bytes, transforms, parameters, generator snapshots and independently reviewed contracts. `scripts/data_integrated_probe.py` executes those fixed public action programs; `scripts/data_integrated_rebind.py` replays committed history through the guarded production bridge with exact physical checkpoints and cold verification. Choose fresh version/output names; do not overwrite evidence.', '',
        'Sources: TopBrain v3 (2026-08-17), containing the normalized TopCoW families, retains its noncommercial/attribution terms and clinical-cohort classification. IXI original images retain CC BY-SA 3.0. The narrowly acquired Zenodo 17393202 version 1.0 vascular reference has separate CC BY 4.0 terms and model-assisted/manually refined provenance. The two annotation joins add zero people. Original archives, CRC/SHA receipts, observed slab coverage and source-specific terms remain on the drive. These are richer source-label research tasks, with no limited-input planner performance claim.', '',
        'Focused Python and Node checks, frontend type-check/build, actual hidden app and independent reviews are recorded in `validation.json`. The generated software control supports drill/CUSA/target/withdrawal and adds zero people. Optional segmentation model attempts failed at local ExFAT/process-IPC export; model prediction itself completed once, but no usable model masks were saved. Source-supported reviewed estimates supply the bounded tasks. The attempts and removed experimental code remain under `anatomy/failed-experiment-code/` and per-case model dispositions.', '',
        'Existing undo/reopen frame-token assertion failure is reproduced with the HEAD bridge and recorded under `qc/task-domain-validation/`; no all-repository pass is claimed. Actual tumor MRI packages with missing rehearsal anatomy retain their previous inspection limits. This cycle introduces no new mass-effect claim, RL training, upload, push or local data cleanup. Previous reversible migration/restoration remains in the original report and manifests.', '',
        '[Machine-readable handoff](handoff.json) · [Independent authority](qc/independent/release-decisions-v1.json) · [Preserved prior report](../catalog/reporting/report-final.md)']
    (destination/'report.md').write_text('\n'.join(lines)+'\n')
    (destination/'START-HERE.md').write_text('# Open the integrated suite\n\nStart with [the capability table and app captures](report.md). Exact canonical package paths and physical replay histories are linked there. [Machine-readable handoff](handoff.json) binds source, review, native and app evidence.\n')
    return document

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root');args=parser.parse_args()
    result=build(args.root);print(json.dumps({'release_id':result['release_id'],'counts':result['counts']}))
