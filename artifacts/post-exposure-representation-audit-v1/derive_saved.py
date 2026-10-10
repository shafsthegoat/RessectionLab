"""Stdlib-only arithmetic over pinned saved metadata. No project/array/model imports."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = Path('build/remind-post-exposure-feasibility-v1/attempt-01')
PINS = {}

def read(path, *, source=False):
    raw = (ROOT/path).read_bytes()
    PINS[str(path)] = {'sha256':hashlib.sha256(raw).hexdigest(), 'bytes':len(raw),
                       'kind':'source_text' if source else 'saved_JSON_metadata'}
    return raw.decode() if source else json.loads(raw)

def write(name, obj):
    (OUT/name).write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')

def main():
    release = read(Path('build/post-exposure-learning-v1/IL64/root-release.json'))
    closure = read(Path('build/post-exposure-learning-v1/IL64/source-index.json'))
    names = ('native_spatial_task','spatial_observations','spatial_policy','public_target_context',
             'public_patient_factory','geometry','patient_planning_learning')
    for name in names:
        path = Path('src/resectionlab')/(name+'.py')
        read(path,source=True)
        assert PINS[str(path)]['sha256'] == closure['source_files'][str(path)]
    architecture = release['learning_protocol']['architecture']
    assert architecture['encoder_channels'] == [8,16] and architecture['ray_samples'] == 5
    assert architecture['physical_reference_mm'] == 10.0
    rows=[]
    for subject in ('002','015','018','045'):
        directory = BASE/('ReMIND-'+subject)
        construction = read(directory/'construction-result.json')
        coverage = read(directory/'initial-public-proposal-coverage.json')
        metrics = read(directory/'episode-metrics.json')
        assert PINS[str(directory/'episode-metrics.json')]['sha256'] == release['inputs']['metrics_ReMIND-'+subject]['sha256']
        assert construction['source_hash'] == metrics['source_hash']
        affine = coverage['native_physical_affine_ras_mm']
        spacing = [math.sqrt(sum(affine[j][i]**2 for j in range(3))) for i in range(3)]
        origin,shape = metrics['crop']['origin_voxels'],metrics['crop']['shape']
        assert shape == [64,64,64] and metrics['crop']['native_geometry_resampled'] is False
        motion = [r for r in metrics['history'] if r['action_id'] != 'STOP']
        counts=[]; inside=[]; distances=[]; cells=set()
        for action in motion:
            values = action['removed_indices_native']
            counts.append(len(values))
            inside.append(sum(all(o <= x < o+n for x,o,n in zip(v,origin,shape)) for v in values))
            cells.update(tuple(v) for v in values)
            distances.append(math.dist(action['entry_mm'],action['tip_mm']))
        assert len(cells) == sum(counts) and counts == inside
        total_target = construction['full_target_extent']['full_region_positive_voxels']
        sampling = construction['target_native_and_NN_sampling']['whole_tumor']
        rows.append({'subject':'ReMIND-'+subject, 'source_hash':metrics['source_hash'],
            'crop_origin_voxels':origin,'crop_shape':shape,'spacing_mm':spacing,
            'crop_cell_extent_mm':[64*s for s in spacing],
            'pooled_bin_cell_extent_mm':[32*s for s in spacing],
            'voxel_volume_mm3':math.prod(spacing), 'full_target_cells':total_target,
            'single_target_cell_global_overlap_fraction':1/total_target,
            'teacher_decisions':len(metrics['history']), 'motion_steps':len(motion),
            'removed_cell_counts_by_motion':counts, 'removed_cells_in_crop_by_motion':inside,
            'single_cell_motion_count':counts.count(1),
            'teacher_entry_to_tip_mm_range':None if not distances else [min(distances),max(distances)],
            'teacher_five_sample_spacing_mm_range':None if not distances else [min(distances)/4,max(distances)/4],
            'initial_emitted_ray_sample_coverage':coverage['ray_sample_coverage_counts'],
            'prior_annotation_sampling':{k:sampling[k] for k in (
                'resampling_can_omit_subvoxel_labels','saved_positive_voxels',
                'source_positive_voxels_from_bound_execution','volume_change_percent')},
            'target_removed_mm3':metrics['target_removed_mm3'],
            'outside_supplied_target_removed_mm3':metrics['normal_removed_mm3']})
    summary = {'scope':'source plus saved TRAIN metadata arithmetic; not a feature/model execution',
        'frozen_learning_head':release['expected_head'], 'architecture':architecture,
        'source_files_match_IL64_release_closure':True,'rows':rows,
        'totals':{'teacher_decisions':sum(r['teacher_decisions'] for r in rows),
            'motion_steps':sum(r['motion_steps'] for r in rows),
            'removed_cells':sum(sum(r['removed_cell_counts_by_motion']) for r in rows),
            'removed_cells_in_crop':sum(sum(r['removed_cells_in_crop_by_motion']) for r in rows),
            'single_cell_motion_steps':sum(r['single_cell_motion_count'] for r in rows)},
        'source_derived_resolution':{'crop_resampling':False,'convolution_stride':1,
            'two_3_cube_convolution_receptive_field_voxels':[5,5,5],
            'maximum_encoded_sites_affected_per_single_input_voxel_per_channel':125,
            'global_pool_bins':[2,2,2],'sites_per_bin_at_64_cube':32768,
            'single_encoded_site_mean_weight':1/32768,
            'maximum_125_site_fraction_of_one_bin':125/32768,
            'adjacent_source_voxel_normalized_coordinate_step':2/63,
            'interpretation':'geometric support and averaging coefficients only; no weights, activation amplitudes or logit attenuation measured'},
        'boundary':{'acquired_image_or_observation_array_reads':0,'checkpoint_reads':0,
            'project_imports':0,'model_forwards':0,'native_calls':0,
            'runtime_or_canonical_source_edits':False,
            'full_actor_feature_aliasing_demonstrated':False,
            'model_compression_bottleneck_established':False}}
    assert summary['totals'] == {'teacher_decisions':29,'motion_steps':25,'removed_cells':109,
                                'removed_cells_in_crop':109,'single_cell_motion_steps':7}
    write('input-pins.json',PINS);write('scalar-calculations.json',summary)
    print(json.dumps(summary['totals']))

if __name__ == '__main__': main()
