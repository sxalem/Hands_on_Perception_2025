#!/usr/bin/env python3
"""
Author: <Salim>
"""

import os, cv2, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from tracker import iou_tracker

# ───────── Dataset Configuration ─────────────────────────────────────
pwd = '/home/salblooshi/Desktop/HOP/iou_kalman_tracker'

DATASETS = {
    'shapes_6dof': {
        'image_dir': '/home/salblooshi/Desktop/HOP/shapes_6dof/images',
        'events_file': '/home/salblooshi/Desktop/HOP/parsed_feature_events.txt',
        'fps': 22.68,
        'output_dir': '/home/salblooshi/Desktop/HOP/results/shapes_6dof'
    },
    'shapes_translation': {
        'image_dir': '/home/salblooshi/Desktop/HOP/shapes_translation/images',
        'events_file': '/home/salblooshi/Desktop/HOP/parsed_feature_events_trans.txt',
        'fps': 22.68,
        'output_dir': '/home/salblooshi/Desktop/HOP/results/shapes_translation'
    },
    'boxes_6dof': {
        'image_dir': '/home/salblooshi/Desktop/HOP/boxes_6dof/images',
        'events_file': '/home/salblooshi/Desktop/HOP/parsed_feature_events_box6dof.txt',
        'fps': 21.74,
        'output_dir': '/home/salblooshi/Desktop/HOP/results/boxes_6dof'
    }
}

# Select dataset to process (change this to process different datasets)
CURRENT_DATASET = 'boxes_6dof'  # 'shapes_6dof', 'shapes_translation', or 'boxes_6dof'

# ───────── Parameters ───────────────────────────────────────────────
# Get dataset-specific parameters
IMAGE_DIR = DATASETS[CURRENT_DATASET]['image_dir']
EVENTS_FILE = DATASETS[CURRENT_DATASET]['events_file']
FPS = DATASETS[CURRENT_DATASET]['fps']
OUTPUT_DIR = DATASETS[CURRENT_DATASET]['output_dir']

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

SKIP_IMAGES = 0
TIME_OFFSET_MS = 0

SPATIAL_RADIUS = 4
TEMPORAL_WINDOW = 5
DENSITY_THRESH = 2
MIN_CLUSTER_SIZE = 5
MAX_CLUSTERS = 1000

SIGMA_IOU = 0.20
T_MIN = 60
MAX_MISSED = 2
TRAIL_LENGTH = 15

SCALE_FACTOR = 10.0
BOX_MIN_SIZE = 10
BOX_MAX_SIZE = 30
BOX_SIZE_DET = 30

WINDOW_MS = 1000.0 / FPS

# ───────── Helper functions ─────────────────────────────────────────
def bbox_iou(b1, b2):
    xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
    xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0.0
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter)

def spatiotemporal_filter(df, r, t_win, thresh):
    if df.empty:
        return df.copy()
    pts = df[['x','y','ts']].to_numpy()
    pts[:,2] /= t_win
    nbrs = NearestNeighbors(radius=r).fit(pts)
    dists, _ = nbrs.radius_neighbors(pts)
    keep = np.array([len(d)-1 for d in dists]) >= thresh
    return df[keep]

def export_tracking_results(track_data, output_file):
    """Export tracking results to a text file.
    Format: frame_idx track_id x y timestamp
    """
    with open(output_file, 'w') as f:
        f.write("# Frame_Index Track_ID X Y Timestamp_ms\n")
        for frame_idx, tid, x, y, ts in track_data:
            f.write(f"{frame_idx} {tid} {x:.2f} {y:.2f} {ts:.2f}\n")
    print(f"Tracking results exported to: {output_file}")

def export_detection_results(det_data, output_file):
    """Export detection results to a text file.
    Format: frame_idx x y
    """
    with open(output_file, 'w') as f:
        f.write("# Frame_Index X Y\n")
        for frame_idx, x, y in det_data:
            f.write(f"{frame_idx} {x:.2f} {y:.2f}\n")
    print(f"Detection results exported to: {output_file}")

# ───────── Load events ──────────────────────────────────────────────
print(f"Processing dataset: {CURRENT_DATASET}")
print(f"Loading events from: {EVENTS_FILE}")

df = pd.read_csv(EVENTS_FILE, header=None,
                 names=['x','y','ts','polarity'])
df['ts'] = (df['ts'] - df['ts'].min()) / 1e6
df = df[df['ts'] >= TIME_OFFSET_MS]
df['frame'] = ((df['ts'] - TIME_OFFSET_MS) // WINDOW_MS).astype(int)
if SKIP_IMAGES:
    df = df[df['frame'] >= SKIP_IMAGES]
    df['frame'] -= SKIP_IMAGES

df = spatiotemporal_filter(df, SPATIAL_RADIUS,
                           TEMPORAL_WINDOW, DENSITY_THRESH)
events_by_frame = df.groupby('frame')[['x','y','ts']]

all_imgs = sorted(f for f in os.listdir(IMAGE_DIR)
                  if f.endswith('.png'))[SKIP_IMAGES:]
if not all_imgs:
    raise RuntimeError("No images found after skipping!")
total_frames = len(all_imgs)

# ───────── Tracker & bookkeeping ────────────────────────────────────
tracker = iou_tracker.VehicleTracker()
missed_counts = {}
trails = defaultdict(list)

det_per_frame = []
track_pos = defaultdict(list)
track_life = {}
iou_scores = defaultdict(list)

# Data collection for export
all_track_data = []  # [(frame_idx, track_id, x, y, timestamp), ...]
all_detection_data = []  # [(frame_idx, x, y), ...]

# ───────── Display window ──────────────────────────────────────────
cv2.namedWindow('Corners+Tracks', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Corners+Tracks', 800, 600)

# ───────── Main loop ───────────────────────────────────────────────
for frame_idx, fname in enumerate(all_imgs):
    t_now = frame_idx * WINDOW_MS

    # 1) get events for this frame
    pts = (events_by_frame.get_group(frame_idx)[['x','y']].to_numpy()
           if frame_idx in events_by_frame.groups else
           np.empty((0,2)))

    # 2) K-Means clustering → centroids
    centroids = []
    if len(pts) >= MIN_CLUSTER_SIZE:
        k = min(max(1, int(np.sqrt(len(pts)))), MAX_CLUSTERS)
        km = KMeans(n_clusters=k, random_state=frame_idx).fit(pts)
        for cid in np.unique(km.labels_):
            if np.sum(km.labels_ == cid) >= MIN_CLUSTER_SIZE:
                centroids.append(km.cluster_centers_[cid])

    # Store detection data for export
    for cx, cy in centroids:
        all_detection_data.append((frame_idx, cx, cy))

    # 3) centroids → detection boxes
    dets = np.array([(int(x-BOX_SIZE_DET/2), int(y-BOX_SIZE_DET/2),
                      int(x+BOX_SIZE_DET/2), int(y+BOX_SIZE_DET/2))
                     for x,y in centroids], dtype=np.int32)
    det_per_frame.append(len(dets))

    # 4) tracker update
    prev_counts = {tr['id']: len(tr['bboxes']) for tr in tracker.Ta}
    tracker.track_iou(dets, t_now, sigma_iou=SIGMA_IOU, t_min=T_MIN)

    # IoU quality bookkeeping
    if len(dets) and tracker.Ta:
        for db in dets:
            best_iou, best_tid = 0., None
            for tr in tracker.Ta:
                tid = tr['id']
                x_pred, _ = tr['kf'].predict_data_association(t_now)
                cx, cy = x_pred.flatten()[:2]

                box = [int(cx - BOX_SIZE_DET/2), int(cy - BOX_SIZE_DET/2),
                       int(cx + BOX_SIZE_DET/2), int(cy + BOX_SIZE_DET/2)]

                iou = bbox_iou(db, box)
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is not None:
                iou_scores[best_tid].append(best_iou)

    # prune missed
    new_Ta = []
    for tr in tracker.Ta:
        tid = tr['id']
        missed_counts[tid] = 0 if len(tr['bboxes']) > prev_counts.get(tid,0) \
                                else missed_counts.get(tid,0) + 1
        if missed_counts[tid] <= MAX_MISSED:
            new_Ta.append(tr)
        else:
            missed_counts.pop(tid, None)
    tracker.Ta = new_Ta

    # life / position stats and export data collection
    for tr in tracker.Ta:
        tid = tr['id']
        x_pred,_ = tr['kf'].predict_data_association(t_now)
        cx, cy = x_pred.flatten()[:2]

        # Store track data for export
        all_track_data.append((frame_idx, tid, cx, cy, t_now))

        if tid not in track_life:
            track_life[tid] = [frame_idx, frame_idx]
        else:
            track_life[tid][1] = frame_idx
        track_pos[tid].append((frame_idx, cx))

    # 5) visualisation
    img = cv2.imread(os.path.join(IMAGE_DIR, fname))
    if img is None: continue

    for x,y in centroids:
        cv2.circle(img, (int(x), int(y)), 2, (0,165,0), -1)

    for tr in tracker.Ta:
        tid = tr['id']
        x_pred,_ = tr['kf'].predict_data_association(t_now)
        cx, cy = x_pred.flatten()[:2]

        last = trails[tid][-1] if trails[tid] else (cx, cy)
        speed = math.hypot(cx-last[0], cy-last[1])
        box = min(max(int(speed*SCALE_FACTOR), BOX_MIN_SIZE), BOX_MAX_SIZE)

        x0,y0 = int(cx-box/2), int(cy-box/2)
        x1,y1 = int(cx+box/2), int(cy+box/2)
        cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),1)
        cv2.putText(img,str(tid),(x0,y0-4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)

        trails[tid].append((int(cx),int(cy)))
        if len(trails[tid]) > TRAIL_LENGTH:
            trails[tid] = trails[tid][-TRAIL_LENGTH:]
        for i in range(1,len(trails[tid])):
            cv2.line(img,trails[tid][i-1],trails[tid][i],
                     (0,255,255),1)

    img_large = cv2.resize(img,None,fx=3.0,fy=3.0,
                           interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Corners+Tracks',img_large)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
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
    f.write("# Track_ID Start_Frame End_Frame Lifespan_Frames\n")
    for tid, life in track_life.items():
        lifespan = life[1] - life[0] + 1
        f.write(f"{tid} {life[0]} {life[1]} {lifespan}\n")
print(f"Track lifespans exported to: {lifespans_output_file}")

# Export summary statistics
summary_file = os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_summary.txt')
with open(summary_file, 'w') as f:
    f.write(f"Dataset: {CURRENT_DATASET}\n")
    f.write(f"Total frames: {total_frames}\n")
    f.write(f"Total tracks: {len(track_life)}\n")
    f.write(f"Average track lifespan: {np.mean([life[1]-life[0]+1 for life in track_life.values()]):.2f} frames\n")
    f.write(f"Total detections: {sum(det_per_frame)}\n")
    f.write(f"Average detections per frame: {np.mean(det_per_frame):.2f}\n")
print(f"Summary exported to: {summary_file}")

# ────────── Stats & plots ───────────────────────────────────────────
# 1. IoU Quality Plot
all_ious = [iou for lst in iou_scores.values() for iou in lst]
plt.figure(figsize=(8,4))
plt.hist(all_ious, bins=50, range=(0,1))
plt.title(f'IoU Quality - {CURRENT_DATASET}')
plt.xlabel('IoU')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_iou_quality.png'))
plt.close()

# 2. Track Lifespan Histogram (X: Frames Survived, Y: Count of Tracks)
plt.figure(figsize=(10,4))
lifespans = [life[1]-life[0]+1 for life in track_life.values()]
if lifespans:
    plt.hist(lifespans, bins=range(1, max(lifespans)+2), edgecolor='black', align='left')
plt.title(f'Histogram: Track Lifespans - {CURRENT_DATASET}')
plt.xlabel('Frames Survived')
plt.ylabel('Number of Track IDs')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'{CURRENT_DATASET}_track_lifespans.png'))
plt.close()

print(f"\nProcessing complete for {CURRENT_DATASET}")
print(f"Results saved to: {OUTPUT_DIR}")
