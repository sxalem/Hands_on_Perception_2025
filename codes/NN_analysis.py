#!/usr/bin/env python3
"""
Nearest-neighbour corner tracker with lifespan + active tracks histograms and data export.

Author: <Salim>
"""

import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# ───────── Dataset Configuration ─────────
pwd = '/home/salblooshi/Desktop/HOP/nn_tracker'

DATASETS = {
    'shapes_6dof': {
        'image_dir': '/home/salblooshi/Desktop/HOP/shapes_6dof/images',
        'events_file': '/home/salblooshi/Desktop/HOP/parsed_feature_events.txt',
        'fps': 22.68
    },
    'shapes_translation': {
        'image_dir': '/home/salblooshi/Desktop/HOP/shapes_translation/images',
        'events_file': '/home/salblooshi/Desktop/HOP/parsed_feature_events_trans.txt',
        'fps': 22.68
    },
    'boxes_6dof': {
        'image_dir': '/home/salblooshi/Desktop/HOP/boxes_6dof/images',
        'events_file': '/home/salblooshi/Desktop/HOP/parsed_feature_events_box6dof.txt',
        'fps': 21.74
    }
}

CURRENT_DATASET = 'boxes_6dof'  # Change this to switch datasets
IMAGE_DIR = DATASETS[CURRENT_DATASET]['image_dir']
EVENTS_FILE = DATASETS[CURRENT_DATASET]['events_file']
FPS = DATASETS[CURRENT_DATASET]['fps']
WINDOW_MS = 1000.0 / FPS
TOTAL_FRAMES = 1298
#TOTAL_FRAMES = 1356

# Tracker parameters
SPATIAL_RADIUS = 4
TEMPORAL_WINDOW = 5
DENSITY_THRESH = 3
MIN_CLUSTER_SIZE = 5
MAX_CLUSTERS = 1000
TRAIL_LENGTH = 15
MAX_MISSED = 2
SKIP_IMAGES = 0
TIME_OFFSET_MS = 0

# ───────── Output Setup ─────────
output_dir = os.path.join(pwd, 'results', CURRENT_DATASET)
os.makedirs(output_dir, exist_ok=True)

def save_plot(fig, name):
    fig.savefig(os.path.join(output_dir, f"{CURRENT_DATASET}_{name}"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# ───────── Load & Filter Events ─────────
df = pd.read_csv(EVENTS_FILE, header=None, names=['x', 'y', 'ts', 'polarity'])
df['ts'] = (df['ts'] - df['ts'].min()) / 1e6
df = df[df['ts'] >= TIME_OFFSET_MS]

def spatiotemporal_filter(df, r, t_win, thresh):
    if df.empty: return df.copy()
    xyz = df[['x', 'y', 'ts']].to_numpy()
    xyz[:, 2] /= t_win
    nbrs = NearestNeighbors(radius=r).fit(xyz)
    dists, _ = nbrs.radius_neighbors(xyz)
    keep = np.array([len(d) - 1 for d in dists]) >= thresh
    return df[keep]

df = spatiotemporal_filter(df, SPATIAL_RADIUS, TEMPORAL_WINDOW, DENSITY_THRESH)
df['frame'] = ((df['ts'] - TIME_OFFSET_MS) // WINDOW_MS).astype(int)
if SKIP_IMAGES:
    df = df[df['frame'] >= SKIP_IMAGES]
    df['frame'] -= SKIP_IMAGES
events_by_frame = df.groupby('frame')[['x', 'y']]

# ───────── Image List ─────────
imgs = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith('.png'))[SKIP_IMAGES:]
if not imgs:
    raise RuntimeError("No images found!")

# ───────── Tracker State ─────────
next_id = 0
active_tracks = {}
missed_counts = defaultdict(int)
trails = defaultdict(list)
track_life = {}
track_count_per_frame = []

# ───────── Main Loop ─────────
for frame_idx, fname in enumerate(imgs):
    pts = events_by_frame.get_group(frame_idx)[['x', 'y']].to_numpy() if frame_idx in events_by_frame.groups else np.empty((0, 2))
    
    centroids = []
    if len(pts) >= MIN_CLUSTER_SIZE:
        k = min(max(1, int(np.sqrt(len(pts)))), MAX_CLUSTERS)
        labels = KMeans(n_clusters=k, random_state=frame_idx).fit(pts).labels_
        for cid in np.unique(labels):
            cpts = pts[labels == cid]
            if len(cpts) >= MIN_CLUSTER_SIZE:
                centroids.append(cpts.mean(axis=0))

    unmatched = set(range(len(centroids)))
    matched_id = set()

    if centroids and active_tracks:
        new_pos = np.array(centroids)
        tids = list(active_tracks.keys())
        prev_pos = np.array([active_tracks[tid] for tid in tids])
        dmat = cdist(new_pos, prev_pos)

        for i, row in enumerate(dmat):
            j = np.argmin(row)
            tid = tids[j]
            speed = np.linalg.norm(np.array(trails[tid][-1]) - np.array(trails[tid][-2])) if len(trails[tid]) >= 2 else 0.0
            thr = max(6, min(30, speed * 1.5))
            if row[j] < thr:
                active_tracks[tid] = centroids[i]
                trails[tid].append(tuple(map(int, centroids[i])))
                trails[tid] = trails[tid][-TRAIL_LENGTH:]
                missed_counts[tid] = 0
                matched_id.add(tid)
                unmatched.discard(i)
                track_life.setdefault(tid, [frame_idx, frame_idx])[1] = frame_idx

    for i in unmatched:
        active_tracks[next_id] = centroids[i]
        trails[next_id] = [tuple(map(int, centroids[i]))]
        missed_counts[next_id] = 0
        track_life[next_id] = [frame_idx, frame_idx]
        next_id += 1

    for tid in list(active_tracks):
        if tid not in matched_id:
            missed_counts[tid] += 1
        if missed_counts[tid] > MAX_MISSED:
            del active_tracks[tid]
            del trails[tid]
            del missed_counts[tid]

    track_count_per_frame.append(len(active_tracks))

# ───────── Stats & Plots ─────────
durations = {tid: (end - start + 1) for tid, (start, end) in track_life.items()}
lifespans = list(durations.values())
avg_active_tracks = np.mean(track_count_per_frame)

print(f"Average Active Tracks per Frame: {avg_active_tracks:.2f}")

# Plot 1: Track Lifespan Histogram
fig1 = plt.figure(figsize=(8, 4))
plt.hist(lifespans, bins=30, range=(0, TOTAL_FRAMES), edgecolor='black')
plt.title('Track Lifespans')
plt.xlabel('Frames')
plt.ylabel('Track Count')
plt.tight_layout()
save_plot(fig1, 'track_lifespans.png')

# Plot 2: Active Tracks per Frame Histogram
fig2 = plt.figure(figsize=(8, 4))
plt.hist(track_count_per_frame, bins=range(0, max(track_count_per_frame)+2), edgecolor='black')
plt.title('Histogram: Active Tracks per Frame')
plt.xlabel('# Active Tracks')
plt.ylabel('Number of Frames')
plt.tight_layout()
save_plot(fig2, 'active_tracks_histogram.png')

# ───────── Export Data ─────────
# Export 1: Track Lifespan TXT
lifespan_txt = os.path.join(output_dir, f"{CURRENT_DATASET}_track_lifespans.txt")
with open(lifespan_txt, 'w') as f:
    f.write("# Track_ID Start_Frame End_Frame Lifespan_Frames\n")
    for tid, (start, end) in track_life.items():
        f.write(f"{tid} {start} {end} {end - start + 1}\n")

# Export 2: Active Tracks per Frame TXT
active_tracks_txt = os.path.join(output_dir, f"{CURRENT_DATASET}_active_tracks_per_frame.txt")
with open(active_tracks_txt, 'w') as f:
    f.write("# Frame_Index Active_Tracks\n")
    for i, count in enumerate(track_count_per_frame):
        f.write(f"{i} {count}\n")

# ───────── Summary ─────────
summary_file = os.path.join(output_dir, f"{CURRENT_DATASET}_summary.txt")
with open(summary_file, 'w') as f:
    f.write(f"Dataset: {CURRENT_DATASET}\n")
    f.write(f"Total frames: {TOTAL_FRAMES}\n")
    f.write(f"Total tracks: {len(durations)}\n")
    f.write(f"Average track lifespan: {np.mean(lifespans):.2f} frames\n")
    f.write(f"Average active tracks/frame: {avg_active_tracks:.2f}\n")

print(f"\n✓ Finished processing {CURRENT_DATASET}")
print(f"→ Outputs saved in: {output_dir}")

