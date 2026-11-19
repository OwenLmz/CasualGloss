#!/usr/bin/env python3
"""Render the structural README overview; this figure contains no measurements."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(15.5, 7.1))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 15.5), ylim=(0, 7.1))
    ax.axis('off')
    ink, muted = '#172b4d', '#42526e'

    def box(x, y, w, h, title, detail='', color='#edf4fc', edge='#547da7'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.10',
                                   facecolor=color, edgecolor=edge, linewidth=1.35))
        center = y + h / 2
        ax.text(x + w / 2, center + (0.19 if detail else 0), title,
                ha='center', va='center', fontsize=11, weight='bold', color=ink,
                linespacing=1.45)
        if detail:
            ax.text(x + w / 2, center - 0.27, detail, ha='center', va='center',
                    fontsize=9.2, color=muted, linespacing=1.45)

    def arrow(start, end, *, style='-', connection='arc3,rad=0', color=muted):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
                                    linewidth=1.45, linestyle=style, color=color,
                                    connectionstyle=connection, shrinkA=3, shrinkB=3))

    ax.text(0.35, 6.78, 'CausalGloss', fontsize=22, weight='bold', color=ink)
    ax.text(0.35, 6.39, 'Action control for a Mandarin-speech-to-pseudo-ASL-gloss cascade',
            fontsize=12, color=muted)
    ax.text(0.35, 5.85, 'INCREMENTAL INFERENCE', fontsize=10.2, weight='bold', color='#476a92')

    box(0.35, 4.30, 1.85, 1.12, 'Mandarin\nspeech')
    box(2.65, 4.30, 2.65, 1.12, 'Streaming translation', 'Existing speech frontend')
    box(5.75, 4.30, 2.30, 1.12, 'English prefix', 'Observed input only')
    box(8.50, 4.19, 3.15, 1.34, 'CausalGloss policy',
        'Qwen2.5-7B + LoRA\nPrefix + committed output history', '#eeeafa', '#7b6baf')
    box(12.10, 4.30, 3.00, 1.12, 'Pseudo-ASL glosses',
        'Committed sequence', '#e9f4ef', '#608f79')
    for start, end in [((2.20, 4.86), (2.65, 4.86)), ((5.30, 4.86), (5.75, 4.86)),
                       ((8.05, 4.86), (8.50, 4.86)), ((11.65, 4.86), (12.10, 4.86))]:
        arrow(start, end)
    ax.text(11.88, 5.70, 'WRITE', ha='center', fontsize=9.5, weight='bold', color=ink)
    ax.text(13.60, 3.95, 'END terminates generation', ha='center', fontsize=9.2, color=muted)
    arrow((9.05, 4.19), (7.20, 4.30), connection='arc3,rad=-0.65', color='#7b6baf')
    ax.text(7.94, 3.38, 'WAIT: read more source', ha='center', fontsize=9.5, color='#63518f')

    ax.text(0.35, 2.70, 'OFFLINE SUPERVISION AND TRAINING', fontsize=10.2,
            weight='bold', color='#476a92')
    box(0.35, 1.03, 3.70, 1.25, 'English transcripts', 'Weak pseudo-gloss targets')
    box(4.70, 1.03, 4.20, 1.25, 'Lexical / semantic evidence', 'WAIT balancing + recovery')
    box(9.55, 1.03, 5.55, 1.25, 'Prefix-level action supervision', 'Two-stage LoRA training')
    arrow((4.05, 1.65), (4.70, 1.65))
    arrow((8.90, 1.65), (9.55, 1.65))
    arrow((10.95, 2.28), (10.95, 4.19), style='--', color='#7b6baf')
    ax.text(11.17, 3.05, 'Trained\nparameters', fontsize=9.3, color='#63518f', linespacing=1.5)
    ax.text(0.35, 0.44, 'Evidence constructs supervision offline. The running policy receives no future source or reference gloss history.',
            fontsize=10, color=muted)
    ax.text(0.35, 0.12, 'Structural overview only. Gloss tokens are not sign-language poses or video.', fontsize=9.3, color=muted)

    fig.subplots_adjust(left=0.015, right=0.995, top=0.99, bottom=0.015)
    destination = Path(__file__).resolve().parent
    fig.savefig(destination / 'system-overview.svg', facecolor='white',
                metadata={'Title': 'CausalGloss supervision and inference',
                          'Description': 'Offline evidence-based supervision trains a policy that reads incremental English and commits pseudo-ASL glosses.'})
    plt.close(fig)


if __name__ == '__main__':
    main()
