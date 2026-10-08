#!/usr/bin/env python3
"""Reproduce a figure from one hash-pinned evaluation JSON; no fitting or image I/O."""
from pathlib import Path
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPORT = ROOT / 'artifacts/resect-case4-sparse-update-v1/fit-freeze/comparison/evaluation.json'
EXPECTED = '928898feaed7b106e4fc3f4e7d280678402f1db24cdc4192cd58d10c3fe0480a'
METHODS = ('no_shift', 'proper_rigid', 'inverse_distance_squared')
NAMES = ('No shift', 'Proper rigid', 'Fixed IDW (1 / distance²)')
COLORS = ('#737b86', '#007f86', '#cf631e')
MARKERS = ('s', 'o', '^')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = REPORT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED, 'Frozen report changed'
    report = json.loads(raw)
    assert report['schema'] == 'conditional-displacement-comparison-v1'
    assert report['units'] == 'mm' and report['validation_landmarks'] == 13
    assert report['confidence_interval'] is None and report['clinical_injury_probability'] is None
    assert report['physical_action_response_validated'] is False
    ids = report['common_supported_ids']
    assert len(ids) == len(set(ids)) == 13 and ids == sorted(ids)
    errors = {}
    fit_ids = None
    for method in METHODS:
        item = report['methods'][method]
        points = item['per_landmark']
        assert [p['row_id'] for p in points] == ids
        assert all(p['status'] == 'supported' for p in points)
        assert not item['excluded_counts'] and item['total_landmarks'] == 13
        errors[method] = np.array([p['error_mm'] for p in points])
        assert np.isfinite(errors[method]).all() and (errors[method] >= 0).all()
        assert item['all_supported'] == item['common_supported']
        fitting = report['B_residuals'][method]
        current_fit_ids = [p['row_id'] for p in fitting['per_landmark']]
        assert fitting['summary']['count'] == len(set(current_fit_ids)) == 6
        assert not set(ids).intersection(current_fit_ids)
        if fit_ids is None: fit_ids = current_fit_ids
        assert fit_ids == current_fit_ids
    # Exactly the saved paired difference with sign reversed for direct IDW−rigid interpretation.
    saved_pair = report['paired_differences']['proper_rigid_minus_inverse_distance_squared']['per_landmark']
    assert [p['row_id'] for p in saved_pair] == ids
    differences = -np.array([p['error_difference_mm'] for p in saved_pair])
    assert np.array_equal(differences, errors[METHODS[2]] - errors[METHODS[1]])
    worse, better = int((differences > 0).sum()), int((differences < 0).sum())

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 11, 'axes.labelsize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(12.6, 8.9))
    ink, muted = '#172839', '#526170'
    fig.text(.065, .948, 'Case 4 · Conditional landmark update', fontsize=22, weight='bold', color=ink)
    fig.text(.065, .912, 'Single DEVELOPMENT patient · observed paired-ultrasound correspondences', fontsize=11.3, color=ink)
    fig.text(.065, .885, '6 fitting observations  |  13 held-out landmarks  |  all points retained', fontsize=10.5, color=muted)

    ax = fig.add_axes([.080, .380, .545, .415])
    paired = fig.add_axes([.748, .380, .197, .415], sharey=ax)
    y = np.arange(len(ids))
    for axis in (ax, paired):
        axis.set_ylim(len(ids)-.50, -.50)
        axis.set_axisbelow(True)
        axis.grid(axis='y', color='#eef0f3', linewidth=.6)
        axis.spines[['top','right']].set_visible(False)
        axis.spines[['bottom','left']].set_color('#c9ced6')
        axis.tick_params(colors=muted, labelsize=9)
    ax.set_title('a  Held-out point error', loc='left', pad=32, color=ink, weight='bold')
    ax.set_yticks(y, [str(i) for i in ids]); ax.set_ylabel('Landmark row ID')
    ax.set_xlim(0, 5.65); ax.set_xticks(np.arange(0, 5.6, 1))
    ax.grid(axis='x', color='#e4e8ed', linewidth=.7)
    ax.set_xlabel('Euclidean error (mm) · lower is better', labelpad=10)
    offsets = (-.16, 0, .16)
    for i in range(len(ids)):
        ax.plot([errors[k][i] for k in METHODS], [y[i]+o for o in offsets],
                color='#d4d9df', linewidth=.8, zorder=1)
    for method, label, color, marker, offset in zip(METHODS, NAMES, COLORS, MARKERS, offsets):
        ax.scatter(errors[method], y+offset, s=34, marker=marker,
                   facecolor=color, edgecolor='white', linewidth=.45, zorder=3, label=label)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), borderaxespad=0, frameon=False,
              ncols=3, fontsize=9, handletextpad=.3, columnspacing=1.15)

    paired.set_title('b  Fixed IDW − rigid', loc='left', pad=32, color=ink, weight='bold')
    paired.text(0, 1.017, f'IDW higher: {worse}/13  ·  lower: {better}/13', transform=paired.transAxes,
                fontsize=9, color=muted, va='bottom')
    paired.axvline(0, color='#9ba5b1', linewidth=.9)
    paired.set_xlim(-.88, .88); paired.set_xticks([-.75, 0, .75], ['−0.75', '0', '+0.75'])
    paired.tick_params(axis='y', labelleft=False, length=0)
    paired.set_xlabel('Error difference (mm)', labelpad=10)
    for i, delta in enumerate(differences):
        color = COLORS[2] if delta > 0 else COLORS[1]
        paired.plot([0, delta], [i, i], color=color, linewidth=1.2)
        paired.scatter([delta], [i], s=24, color=color, zorder=3)
        paired.text(delta + (.045 if delta >= 0 else -.045), i, f'{delta:+.2f}',
                    ha='left' if delta >= 0 else 'right', va='center', fontsize=8, color=color)
    paired.text(.5, -.16, '← IDW lower     IDW higher →', transform=paired.transAxes,
                ha='center', va='top', fontsize=8.5, color=muted)

    # All summaries are copied from the frozen report; fitting residuals stay explicitly separate.
    table_ax = fig.add_axes([.08, .135, .865, .115]); table_ax.set_axis_off()
    table_ax.text(0, 1.08, 'Frozen-report summaries (mm)', fontsize=10.2, weight='bold', color=ink)
    columns = ['', 'Held-out RMS', 'Held-out median', 'Held-out maximum', 'Fit RMS (6 only)']
    rows = []
    for method, name in zip(METHODS, NAMES):
        summary = report['methods'][method]['common_supported']
        fit_rms = report['B_residuals'][method]['summary']['rms_mm']
        fit_text = '< 1e−15' if 0 < fit_rms < 1e-15 else f'{fit_rms:.3f}'
        rows.append([name, f'{summary["rms_mm"]:.3f}', f'{summary["median_mm"]:.3f}',
                     f'{summary["maximum_mm"]:.3f}', fit_text])
    table = table_ax.table(cellText=rows, colLabels=columns, loc='upper left', cellLoc='center',
                           colWidths=[.275,.175,.185,.19,.175], bbox=[0, -.015, 1, .99])
    table.auto_set_font_size(False); table.set_fontsize(9)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor('white'); cell.set_linewidth(1)
        cell.set_facecolor('#eaf0f3' if r==0 else ('#f4f6f8' if r%2 else 'white'))
        if r==0: cell.set_text_props(weight='bold', color=ink)
        elif c==0: cell.set_text_props(ha='left', color=COLORS[r-1])
        else: cell.set_text_props(color=ink)
    fig.text(.08, .100, 'Fit residuals describe the conditioning observations, not validation. No confidence interval is reported.',
             fontsize=8.5, color=muted)
    fig.text(.08, .070, 'Scope: conditional update using observed correspondences. This is not a preoperative-only prediction, RL result,', fontsize=8.7, color=ink)
    fig.text(.08, .052, 'dense-mechanics validation or clinical-safety claim. Global mathematical support does not verify retained tissue.', fontsize=8.7, color=ink)
    fig.text(.08, .025, f'Frozen evaluation SHA256: {EXPECTED[:16]}…  |  Source-frame errors: RAS+, mm  |  Descriptive single-case evidence',
             fontsize=7.5, color=muted)

    png = OUT/'case4-held-out-landmark-errors.png'; pdf = OUT/'case4-held-out-landmark-errors.pdf'
    fig.savefig(png, dpi=220, metadata={'Software': 'RessectionLab frozen Case4 report plot'})
    fig.savefig(pdf, metadata={'Title': 'Case4 conditional landmark update: single development patient',
                               'Subject': 'All 13 held-out point errors from frozen evaluation; no new fitting',
                               'Creator': 'RessectionLab plot.py', 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    assert digest(REPORT) == EXPECTED
    provenance = {'schema': 'resect-case4-frozen-error-figure.v1',
        'source': {'path': str(REPORT.relative_to(ROOT)), 'sha256': EXPECTED, 'bytes': len(raw)},
        'source_unchanged_after_plot': True,
        'script': {'path': str(Path(__file__).relative_to(ROOT)), 'sha256': digest(Path(__file__))},
        'versions': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__, 'numpy': np.__version__},
        'dataset_role': 'single DEVELOPMENT patient', 'fitting_row_ids': fit_ids, 'held_out_row_ids': ids,
        'method_keys': list(METHODS), 'display_names': list(NAMES),
        'all_plotted_point_errors_mm': {m: errors[m].tolist() for m in METHODS},
        'paired_difference': {'definition': 'fixed IDW error minus proper rigid error; positive means IDW worse',
                              'values_mm': differences.tolist(), 'idw_higher_count': worse, 'idw_lower_count': better},
        'excluded_points': 0, 'new_fitting': False, 'original_patient_tags_or_images_read': False,
        'confidence_interval': None,
        'limits': ['Observed paired-US conditional update; not preoperative-only or RL.',
                   'No dense-mechanics or clinical-safety validation.',
                   'Global mathematical support is not verified retained tissue.',
                   'Six fitting residuals are conditioning diagnostics, not validation.'],
        'artifacts': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size} for p in (png,pdf)}}
    (OUT/'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': 'rendered_from_frozen_report', 'source_sha256': EXPECTED,
                      'artifacts': provenance['artifacts'], 'idw_higher_count': worse, 'idw_lower_count': better}, indent=2))


if __name__ == '__main__':
    main()
