import pandas as pd
import matplotlib.pyplot as plt

# ────────────── File Paths ──────────────
lifespan_files = {
    'Shapes 6DoF': '/home/salblooshi/Desktop/HOP/nn_tracker/results/shapes_6dof/shapes_6dof_track_lifespans.txt',
    'Shapes Translation': '/home/salblooshi/Desktop/HOP/nn_tracker/results/shapes_translation/shapes_translation_track_lifespans.txt',
    'Boxes 6DoF': '/home/salblooshi/Desktop/HOP/nn_tracker/results/boxes_6dof/boxes_6dof_track_lifespans.txt'
}

active_files = {
    'Shapes 6DoF': '/home/salblooshi/Desktop/HOP/nn_tracker/results/shapes_6dof/shapes_6dof_active_tracks_per_frame.txt',
    'Shapes Translation': '/home/salblooshi/Desktop/HOP/nn_tracker/results/shapes_translation/shapes_translation_active_tracks_per_frame.txt',
    'Boxes 6DoF': '/home/salblooshi/Desktop/HOP/nn_tracker/results/boxes_6dof/boxes_6dof_active_tracks_per_frame.txt'
}

# ────────────── Plot 1: Track Lifespans ──────────────
plt.figure(figsize=(12, 6))
for label, path in lifespan_files.items():
    df = pd.read_csv(path, comment='#', delim_whitespace=True, names=['ID', 'Start', 'End', 'Lifespan'])
    plt.hist(df['Lifespan'], bins=range(1, df['Lifespan'].max() + 2), histtype='step', linewidth=2, label=label)
plt.title('Track Lifespans per Dataset')
plt.xlabel('Frames Survived')
plt.ylabel('Track Count')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/salblooshi/Desktop/HOP/nn_tracker/results/comparison_track_lifespans.png')
plt.show()

# ────────────── Plot 2: Active Tracks per Frame ──────────────
plt.figure(figsize=(12, 6))
for label, path in active_files.items():
    df = pd.read_csv(path, comment='#', delim_whitespace=True, names=['Frame', 'Active'])
    plt.hist(df['Active'], bins=range(df['Active'].min(), df['Active'].max() + 2), histtype='step', linewidth=2, label=label)
plt.title('Histogram: Active Tracks per Frame (All Datasets)')
plt.xlabel('# Active Tracks')
plt.ylabel('Number of Frames')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/salblooshi/Desktop/HOP/nn_tracker/results/comparison_active_tracks_per_frame.png')
plt.show()
