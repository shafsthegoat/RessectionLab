#!/usr/bin/env python3
"""Plot the frozen donor-transfer result; no source acquisition or model fit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'testcases') or args.output.exists():
        parser.error('Choose a new PNG under testcases/')
    result = json.loads(args.result.read_text())
    prep = json.loads(args.preparation.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5))
    fig.suptitle('One donor-held-out motor-cortex specimen: mixed force agreement',
                 x=.06, y=.98, ha='left', fontsize=15, fontweight='bold')
    names = ['compression_c3', 'tension_c3']
    for ax, name, title in zip(axes[0], names, ['Compression', 'Tension']):
        data = result['results'][name]
        x = np.asarray(data['displacementMm'])
        order = np.argsort(x)
        y = np.asarray(data['observedForceN']) * 1000
        all_mesh = np.array([p['models']['continuum']['forceN'] for p in data['predictions']]) * 1000
        ax.fill_between(x[order], all_mesh.min(axis=0)[order], all_mesh.max(axis=0)[order],
                        color='#277b96', alpha=.2, label='Prespecified mesh range')
        ax.plot(x[order], all_mesh[-1, order], color='#277b96', label='Calibrated continuum')
        ax.scatter(x, y, color='#192f43', s=35, label='Acquired force', zorder=5)
        ax.axhline(0, color='#7b858f', linestyle=':', label='Zero-force baseline')
        ax.set(title=f'{title}: 4 original points', xlabel='Released displacement (mm)', ylabel='Force (mN)')
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8, loc='best')
    colors = ['#277b96', '#b64a43', '#d39743', '#7b858f']
    model_names = ['continuum', 'legacyDefault', 'legacyMatchedStrain', 'zeroForce']
    labels = ['Continuum', 'Legacy default', 'Legacy equal strain', 'Zero force']
    ax = axes[1, 0]
    for i, (model, color) in enumerate(zip(model_names, colors)):
        values = [result['results'][n]['predictions'][-1]['models'][model]['forceRmseN'] * 1000 for n in names]
        ax.bar(np.arange(2) + (i-1.5)*.19, values, width=.18, color=color, label=labels[i])
    ax.set_xticks([0, 1], ['Compression', 'Tension'])
    ax.set(title='Finest-mesh error, including both legacy baselines', ylabel='Force RMSE (mN)')
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(axis='y', alpha=.15)
    ax = axes[1, 1]
    for name, label, color in [('development', 'Development fixture', '#277b96'),
                                ('evaluation', 'Evaluation fixture', '#8a5b99')]:
        rows = prep['fixtures'][name]
        values = np.array([r['continuumSlopeNPerMmPerPa'] for r in rows])
        ax.plot([r['xyCount'] for r in rows], values/values[-1], 'o-', label=label, color=color)
    ax.axhline(1, color='#7b858f', linewidth=.8)
    ax.set(title='Continuum mesh response (unit shear scale)', xlabel='Cross-section voxel count',
           ylabel='Reaction slope / finest slope', xticks=[12, 20, 28])
    ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.text(.06, .025,
             'HBE_02_03, processed c3 curves; no force-zero correction. Tension zero baseline beats continuum.\n'
             'Mesh range is numerical sensitivity, not measurement uncertainty. Patient mechanics and physiological RL remain on hold.',
             fontsize=9, color='#36495a')
    fig.tight_layout(rect=[.035, .09, 1, .94], h_pad=2.0, w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
