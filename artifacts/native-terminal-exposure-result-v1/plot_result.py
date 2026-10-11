"""Plot all six fixed fine-grid rows from a completed accepted comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    if report['status'] != 'completed' or not report['useful_criterion']['meets_predeclared_useful_criterion']:
        raise ValueError('This plot requires the complete accepted comparison')
    rows = [r for r in report['rows'] if r['spacing_mm'] == .25]
    if len(rows) != 6 or not all(r['native']['complete_sequence'] for r in rows):
        raise ValueError('Keep all six complete fine-grid scenes')
    before, after = [], []
    for row in rows:
        cap = row['exact_continuous_target_cap_mm3']
        before.append(100*(1-row['baseline_comparison']['baseline_physical_target_mm3']/cap))
        after.append(100*(1-row['partial_evacuation']['physical_target_evacuated_mm3']/cap))
    labels = ['Aligned\nsuction', 'Aligned\naspirator', 'Half-phase\nsuction',
              'Half-phase\naspirator', 'Tilted\nsuction', 'Tilted\naspirator']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(10, 5.3), dpi=160)
    fig.subplots_adjust(left=.085, right=.985, bottom=.22, top=.70)
    fig.text(.085, .93, 'Target-volume error after the exposure-timing fix',
             fontsize=17, fontweight='bold', color='#102c44')
    fig.text(.085, .87, 'Generated geometry · same tools, actions and resolution · lower is better',
             fontsize=10, color='#526675')
    old = ax.bar([i-.19 for i in range(6)], before, .36, color='#afbac5', label='Retained baseline')
    new = ax.bar([i+.19 for i in range(6)], after, .36, color='#176f91', label='Terminal exposure closure')
    ax.bar_label(old, fmt='%.1f%%', fontsize=9, padding=4, color='#526675')
    ax.bar_label(new, fmt='%.1f%%', fontsize=9, padding=4, color='#164966')
    ax.set_xticks(range(6), labels)
    ax.set_ylabel('Uncredited reference target volume (%)')
    ax.set_ylim(0, max(before+after)*1.20)
    ax.grid(axis='y', color='#e3e8ec', linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#c8d0d6')
    ax.tick_params(axis='both', length=0, pad=8)
    fig.legend(handles=[old, new], loc='upper left', bbox_to_anchor=(.075, .83),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.085, .065, 'Reference: exact executed tool cap. Independently checked geometry; no patient or force validation.',
             fontsize=9, color='#526675')
    fig.savefig(args.output, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
