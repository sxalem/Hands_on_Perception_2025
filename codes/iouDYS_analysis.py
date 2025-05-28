#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocity-adaptive event-cluster tracker 
• variable box size   : 10 px → 30 px  (based on average speed)
• variable slice time : 16.6 ms → 166.6 ms (based on average speed)
• Exports tracking results for comparison with fixed sampling
"""

# ───────────────────────────── imports ──────────────────────────────
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans

# tracker package
from tracker import iou_tracker

# ───────── Dataset Configuration ─────────────────────────────────────
DATASETS = {
    'shapes_6dof': {
        'events_file': '/home/salblooshi/Desktop/HOP/parsed_feature_events.txt',
        'output_dir': '/home/salblooshi/Desktop/HOP/results/shapes_6dof'
    },
    'shapes_translation': {
        'events_file': '/home/salblooshi/Desktop/HOP/parsed_feature_events_trans.txt',
        'output_dir': '/home/salblooshi/Desktop/HOP/results/shapes_translation'
    },
    'boxes_6dof': {
        'events_file': '/home/salblooshi/Desktop/HOP/parsed_feature_events_box6dof.txt',
        'output_dir': '/home/salblooshi/Desktop/HOP/results/boxes_6dof'
    }
}

# Select dataset to process
CURRENT_DATASET = 'boxes_6dof'  # 'shapes_6dof', 'shapes_translation', or 'boxes_6dof'

# Get dataset-specific parameters
EVENTS_FILE = DATASETS[CURRENT_DATASET]['events_file']
OUTPUT_DIR = DATASETS[CURRENT_DATASET]['output_dir']

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────── canvas parameters ──────────────────────
WIDTH, HEIGHT = 240, 180
FPS = 30.0

# ───────────── tracker, filter & clustering parameters ─────────────
SPATIAL_RADIUS   = 4
TEMPORAL_WINDOW  = 5
DENSITY_THRESH   = 2
MIN_CLUSTER_SIZE = 5                  #thats my settings for boxes_6dof sata sets becuase its very noicy , I have to change it for other data sets
MAX_CLUSTERS     = 1000

T_MIN      = 60
MAX_MISSED = 2
TRAIL_LEN  = 15

# ───────────── speed-→-box-size / slice-length mapping ─────────────
MIN_BOX_SIZE = 10
MAX_BOX_SIZE = 30
V_MIN        = 0.0
V_MAX        = 30.0

def box_size_from_speed(v,
                        v_min=V_MIN, v_max=V_MAX,
                        s_min=MIN_BOX_SIZE, s_max=MAX_BOX_SIZE) -> int:
    """Linear clamp from speed to box side length (px)."""
    alpha = np.clip((v - v_min) / (v_max - v_min), 0.0, 1.0)
    return int(round(s_min + alpha * (s_max - s_min)))

def slice_duration(v,
                   v_min=V_MIN, v_max=V_MAX,
                   t_min=16.6, t_max=166.6) -> float:
    """
    Linear clamp: fast → short window.
    Returns duration in milliseconds.
    """
    alpha = np.clip((v - v_min) / (v_max - v_min), 0.0, 1.0)
    return t_max - alpha * (t_max - t_min)

# ──────────────────────── Export functions ──────────────────────────────
def export_tracking_results(track_data, output_file):
    """Export tracking results to a text file.
    Format: slice_idx track_id x y timestamp
    """
    with open(output_file, 'w') as f:
        f.write("# Slice_Index Track_ID X Y Timestamp_ms\n")
        for slice_idx, tid, x, y, ts in track_data:
            f.write(f"{slice_idx} {tid} {x:.2f} {y:.2f} {ts:.2f}\n")
    print(f"Tracking results exported to: {output_file}")

def export_detection_results(det_data, output_file):
    """Export detection results to a text file.
    Format: slice_idx x y
    """
    with open(output_file, 'w') as f:
        f.write("# Slice_Index X Y\n")
        for slice_idx, x, y in det_data:
            f.write(f"{slice_idx} {x:.2f} {y:.2f}\n")
    print(f"Detection results exported to: {output_file}")

def export_dynamic_params(params_data, output_file):
    """Export dynamic parameters (box size, slice duration) to a text file."""
    with open(output_file, 'w') as f:
        f.write("# Slice_Index Avg_Speed Box_Size Slice_Duration_ms\n")
        for slice_idx, avg_speed, box_size, duration in params_data:
            f.write(f"{slice_idx} {avg_speed:.2f} {box_size} {duration:.2f}\n")
    print(f"Dynamic parameters exported to: {output_file}")

def density_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps events that lie in spatio-temporal dense regions."""
    if df.empty:
        return df
    xyz = df[['x', 'y', 'ts']].to_numpy()
    xyz[:, 2] /= TEMPORAL_WINDOW
    nbr  = NearestNeighbors(radius=SPATIAL_RADIUS).fit(xyz)
    keep = np.array([len(v) - 1 for v in nbr.radius_neighbors(xyz)[0]]) \
           >= DENSITY_THRESH
    return df[keep]

# ──────────────────────── load events ──────────────────────────────
print(f"▸ Processing dataset: {CURRENT_DATASET}")
print(f"▸ Loading events from: {EVENTS_FILE}")

ev = pd.read_csv(EVENTS_FILE, header=None, names=['x', 'y', 'ts', 'pol'])
ev['ts'] = (ev['ts'] - ev['ts'].min()) * 1e-6        # μs → ms
ev.sort_values('ts', inplace=True, ignore_index=True)

# normalise coordinates to canvas
x_scale = WIDTH  / ev['x'].max() if ev['x'].max() > 0 else 1.0
y_scale = HEIGHT / ev['y'].max() if ev['y'].max() > 0 else 1.0
ev['x'] *= x_scale
ev['y'] *= y_scale
print(f"▸ Scaled events to {WIDTH}×{HEIGHT} canvas")

# ───────────────────── initialise objects ─────────────────────────
tracker = iou_tracker.VehicleTracker()

trails        = defaultdict(list)
missed        = defaultdict(int)
track_pos     = defaultdict(list)
track_life    = defaultdict(lambda: [None, None])
iou_scores    = defaultdict(list)

video_frames  = []
cluster_counts= []
track_counts  = []
slice_durations= []

# Data collection for export
all_track_data = []
all_detection_data = []
all_dynamic_params = []

# ───────────────────── main processing loop ───────────────────────
start_time = ev['ts'].iloc[0]
end_time   = ev['ts'].iloc[-1]
slice_idx  = 0

cv2.namedWindow('Clustered Points Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Clustered Points Tracking', 800, 600)
print("▸ Processing slices …  (press 'q' to abort)")

while start_time < end_time:

    # —— A. average speed of all active Kalman filters ————————
    speeds = []
    for tr in tracker.Ta:
        d     = tr['kf'].motion_model.dims
        v_vec = tr['kf'].motion_model.x[d:2*d].flatten()
        speeds.append(np.linalg.norm(v_vec))
    avg_speed = float(np.mean(speeds)) if speeds else 0.0

    # —— B. derive slice length & box size ————————————————
    dt       = slice_duration(avg_speed)
    box_size = box_size_from_speed(avg_speed)
    slice_durations.append(dt)
    
    # Store dynamic parameters
    all_dynamic_params.append((slice_idx, avg_speed, box_size, dt))

    window_end = start_time + dt

    # —— C. collect and filter events ————————————————
    ev_slice = ev[(ev['ts'] >= start_time) & (ev['ts'] < window_end)]
    if not ev_slice.empty:
        ev_slice = density_filter(ev_slice)

    # —— D. cluster to centroids ————————————————
    pts       = ev_slice[['x', 'y']].to_numpy() if not ev_slice.empty \
                else np.empty((0, 2))
    centroids = []
    if len(pts) >= MIN_CLUSTER_SIZE:
        k  = int(np.clip(np.sqrt(len(pts)), 1, MAX_CLUSTERS))
        km = MiniBatchKMeans(n_clusters=k, batch_size=64,
                             random_state=0).fit(pts)
        for cid in np.unique(km.labels_):
            if np.sum(km.labels_ == cid) >= MIN_CLUSTER_SIZE:
                centroids.append(km.cluster_centers_[cid])

    # Store detection data
    for cx, cy in centroids:
        all_detection_data.append((slice_idx, cx, cy))

    # —— E. detections for IOU tracker (adaptive box size) ————
    half = box_size // 2
    dets = np.array([
        (int(x - half), int(y - half), int(x + half), int(y + half))
        for x, y in centroids
    ])

    # —— F. tracker update ———————————————————————————————
    ts = ev_slice['ts'].mean() if not ev_slice.empty else window_end
    sizes_before = {tr['id']: len(tr['bboxes']) for tr in tracker.Ta}
    tracker.track_iou(dets, ts, sigma_iou=0.2, t_min=T_MIN)

    # bookkeeping
    active_tracks = {}
    for tr in tracker.Ta:
        tid = tr['id']

        # miss counter
        if len(tr['bboxes']) > sizes_before.get(tid, 0):
            missed[tid] = 0
        else:
            missed[tid] += 1

        # life span
        if track_life[tid][0] is None:
            track_life[tid][0] = slice_idx
        track_life[tid][1] = slice_idx

        # predicted centre
        cx, cy = tr['kf'].predict_data_association(ts)[0].flatten()[:2]
        if not np.isnan(cx) and missed[tid] <= MAX_MISSED:
            active_tracks[tid] = (cx, cy)
            trails[tid].append((int(cx), int(cy)))
            trails[tid] = trails[tid][-TRAIL_LEN:]
            track_pos[tid].append((slice_idx, cx))
            
            # Store track data for export
            all_track_data.append((slice_idx, tid, cx, cy, ts))

        # IoU quality
        if len(tr['bboxes']) >= 2:
            a, b = tr['bboxes'][-2], tr['bboxes'][-1]
            inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * \
                    max(0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1]) + \
                    (b[2]-b[0])*(b[3]-b[1]) - inter
            if union > 0:
                iou_scores[tid].append(inter / union)

    # prune
    tracker.Ta = [tr for tr in tracker.Ta if missed[tr['id']] <= MAX_MISSED]

    # —— G. draw frame ————————————————————————————————
    img = np.ones((HEIGHT, WIDTH, 3), np.uint8) * 255

    # trails
    for tid, pts in trails.items():
        if tid in active_tracks and len(pts) > 1:
            for i in range(1, len(pts)):
                cv2.line(img, pts[i-1], pts[i], (255, 128, 0), 2)

    # centroids
    for x, y in centroids:
        cv2.circle(img, (int(x), int(y)), 3, (0, 200, 0), -1)

    # boxes & IDs
    for tid, (cx, cy) in active_tracks.items():
        x0, y0 = int(cx - half), int(cy - half)
        x1, y1 = int(cx + half), int(cy + half)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img, str(tid), (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # overlay stats
    cv2.putText(img,
                f"Slice {slice_idx}  |  clusters {len(centroids)}  |  tracks"
                f" {len(active_tracks)}",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(img,
                f"v̄={avg_speed:4.1f}px   box={box_size}px   Δt={dt:5.1f}ms",
                (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    video_frames.append(img.copy())
    cluster_counts.append(len(centroids))
    track_counts.append(len(active_tracks))

    # show
    cv2.imshow('Clustered Points Tracking', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("¶ user abort")
        break

    # next slice
    start_time = window_end
    slice_idx += 1

cv2.destroyAllWindows()

# ───────── Export Results ───────────────────────────────────────────
# Export tracking results
track_output_file = os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_tracking_results.txt')
export_tracking_results(all_track_data, track_output_file)

# Export detection results
det_output_file = os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_detection_results.txt')
export_detection_results(all_detection_data, det_output_file)

# Export IoU scores
iou_output_file = os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_iou_scores.txt')
with open(iou_output_file, 'w') as f:
    f.write("# Track_ID IoU_Score\n")
    for tid, scores in iou_scores.items():
        for score in scores:
            f.write(f"{tid} {score:.4f}\n")
print(f"IoU scores exported to: {iou_output_file}")

# Export track lifespans
lifespans_output_file = os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_track_lifespans.txt')
with open(lifespans_output_file, 'w') as f:
    f.write("# Track_ID Start_Slice End_Slice Lifespan_Slices\n")
    for tid, (start, end) in track_life.items():
        if start is not None:
            lifespan = end - start + 1
            f.write(f"{tid} {start} {end} {lifespan}\n")
print(f"Track lifespans exported to: {lifespans_output_file}")

# Export dynamic parameters
params_output_file = os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_dynamic_params.txt')
export_dynamic_params(all_dynamic_params, params_output_file)

# Export summary statistics
summary_file = os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_summary.txt')
with open(summary_file, 'w') as f:
    f.write(f"Dataset: {CURRENT_DATASET}\n")
    f.write(f"Total slices: {slice_idx}\n")
    f.write(f"Total tracks: {len(track_life)}\n")
    valid_lifespans = [(end - start + 1) for start, end in track_life.values() if start is not None]
    f.write(f"Average track lifespan: {np.mean(valid_lifespans):.2f} slices\n")
    f.write(f"Total detections: {len(all_detection_data)}\n")
    f.write(f"Average detections per slice: {np.mean(cluster_counts):.2f}\n")
    f.write(f"Average slice duration: {np.mean(slice_durations):.2f} ms\n")
    f.write(f"Min slice duration: {np.min(slice_durations):.2f} ms\n")
    f.write(f"Max slice duration: {np.max(slice_durations):.2f} ms\n")
print(f"Summary exported to: {summary_file}")

# ──────────────────────── PLOTS & SAVING ───────────────────────────
print("▸ Creating plots …")

# 1. IoU quality histogram
all_iou = [v for lst in iou_scores.values() for v in lst]
if all_iou:
    fig = plt.figure(figsize=(8, 4))
    plt.hist(all_iou, bins=50, range=(0, 1))
    plt.title(f'IoU Quality Distribution - {CURRENT_DATASET} (Dynamic)')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_iou_quality.png'))
    plt.close()

# 2. Track lifespan histogram
valid_lifespans = [(end - start + 1) for start, end in track_life.values() if start is not None]
if valid_lifespans:
    fig = plt.figure(figsize=(10, 4))
    plt.hist(valid_lifespans, bins=range(1, max(valid_lifespans)+2), edgecolor='black', align='left')
    plt.title(f'Track Lifespans - {CURRENT_DATASET} (Dynamic)')
    plt.xlabel('Slices Survived')
    plt.ylabel('Number of Track IDs')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_track_lifespans.png'))
    plt.close()

# 3. Slice duration over time
fig = plt.figure(figsize=(10, 4))
plt.plot(slice_durations, 'g-', linewidth=2)
plt.title(f'Slice Duration (Velocity Adaptive) - {CURRENT_DATASET}')
plt.xlabel('Slice Index')
plt.ylabel('Duration (ms)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_slice_durations.png'))
plt.close()

# 4. Detections per slice
fig = plt.figure(figsize=(10, 4))
plt.plot(cluster_counts, 'b-', linewidth=2)
plt.title(f'Detections per Slice - {CURRENT_DATASET} (Dynamic)')
plt.xlabel('Slice Index')
plt.ylabel('Number of Detections')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_detections_per_slice.png'))
plt.close()

# 5. Active tracks over time
fig = plt.figure(figsize=(10, 4))
plt.plot(track_counts, 'r-', linewidth=2)
plt.title(f'Active Tracks Over Time - {CURRENT_DATASET} (Dynamic)')
plt.xlabel('Slice Index')
plt.ylabel('Number of Active Tracks')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_active_tracks.png'))
plt.close()

print(f"\n▸ Processing complete for {CURRENT_DATASET} (Dynamic Sampling)")
print(f"  Results saved to: {OUTPUT_DIR}")
print(f"  Slices processed: {slice_idx}")
print(f"  Average Δt: {np.mean(slice_durations):.1f} ms  "
      f"(min {np.min(slice_durations):.1f} / max {np.max(slice_durations):.1f})")
