#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Velocity–adaptive event-cluster tracker
• variable box size   : 10 px → 30 px  (based on average speed)
• variable slice time : 16.6 ms → 166.6 ms (based on average speed)
Everything else is identical to the original reference script.
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

# tracker package (make sure it is import-able)
from tracker import iou_tracker                 # VehicleTracker

# ─────────────────── file / canvas parameters ──────────────────────
EVENTS_FILE = (
    '/home/salblooshi/Desktop/Master/Term2/Preception/Project2/'
    'corner_workspace/src/parsed_feature_events_box6dof.txt'
)

WIDTH, HEIGHT = 240, 180                       # fixed output resolution
FPS = 30.0                                     # video FPS

# ───────────── tracker, filter & clustering parameters ─────────────
SPATIAL_RADIUS   = 4
TEMPORAL_WINDOW  = 5
DENSITY_THRESH   = 10
MIN_CLUSTER_SIZE = 5
MAX_CLUSTERS     = 200
                                                       #thats my settings for boxes_6dof sata sets becuase its very noicy , I have to change it for other data sets
T_MIN      = 60
MAX_MISSED = 3
TRAIL_LEN  = 15

# ───────────── speed-→-box-size / slice-length mapping ─────────────
MIN_BOX_SIZE = 10                              # px
MAX_BOX_SIZE = 30                              # px
V_MIN        = 0.0                             # px / slice
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

# ──────────────────────── I/O helpers ──────────────────────────────
def save_outputs(video_frames, plots, output_dir='shapes_6dof') -> None:
    """Write AVI and PNGs to a sibling folder next to the script."""
    here = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(here, output_dir)
    os.makedirs(save_path, exist_ok=True)

    # video
    if video_frames:
        vid_path = os.path.join(save_path, 'clustered_visualization.avi')
        fourcc   = cv2.VideoWriter_fourcc(*'XVID')
        out      = cv2.VideoWriter(vid_path, fourcc, FPS, (WIDTH, HEIGHT))
        if not out.isOpened():                 # fallback codec
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out    = cv2.VideoWriter(vid_path, fourcc, FPS, (WIDTH, HEIGHT))

        for fr in video_frames:
            out.write(fr)
        out.release()
        print(f"[✓] Video  : {vid_path}")

    # plots
    names = [
        'x_vs_frame_top20.png', 'detections_per_frame.png',
        'iou_quality.png',      'active_tracks.png',
        'slice_durations.png'
    ]
    for fig, name in zip(plots, names):
        p = os.path.join(save_path, name)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"[✓] Plot   : {p}")

def density_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keeps events that lie in spatio-temporal dense regions."""
    if df.empty:
        return df
    xyz = df[['x', 'y', 'ts']].to_numpy()
    xyz[:, 2] /= TEMPORAL_WINDOW              # scale time
    nbr  = NearestNeighbors(radius=SPATIAL_RADIUS).fit(xyz)
    keep = np.array([len(v) - 1 for v in nbr.radius_neighbors(xyz)[0]]) \
           >= DENSITY_THRESH
    return df[keep]

# ──────────────────────── load events ──────────────────────────────
print("▸ Loading events …")
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

# ───────────────────── main processing loop ───────────────────────
start_time = ev['ts'].iloc[0]
end_time   = ev['ts'].iloc[-1]
slice_idx  = 0

cv2.namedWindow('Clustered Points Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Clustered Points Tracking', 800, 600)
print("▸ Processing slices …  (press ‘q’ to abort)")

while start_time < end_time:

    # —— A. average speed of all active Kalman filters 
    speeds = []
    for tr in tracker.Ta:
        d     = tr['kf'].motion_model.dims
        v_vec = tr['kf'].motion_model.x[d:2*d].flatten()
        speeds.append(np.linalg.norm(v_vec))
    avg_speed = float(np.mean(speeds)) if speeds else 0.0

    # —— B. derive slice length & box size
    dt       = slice_duration(avg_speed)       # ms
    box_size = box_size_from_speed(avg_speed)  # px
    slice_durations.append(dt)

    window_end = start_time + dt

    # —— C. collect and filter events
    ev_slice = ev[(ev['ts'] >= start_time) & (ev['ts'] < window_end)]
    if not ev_slice.empty:
        ev_slice = density_filter(ev_slice)

    # —— D. cluster to centroids
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

    # —— E. detections for IOU tracker (adaptive box size) 
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
            track_life[tid][0] = slice_idx          # first seen
        track_life[tid][1] = slice_idx              # last seen

        # predicted centre
        cx, cy = tr['kf'].predict_data_association(ts)[0].flatten()[:2]
        if not np.isnan(cx) and missed[tid] <= MAX_MISSED:
            active_tracks[tid] = (cx, cy)
            trails[tid].append((int(cx), int(cy)))
            trails[tid] = trails[tid][-TRAIL_LEN:]
            track_pos[tid].append((slice_idx, cx))

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

# ──────────────────────── PLOTS & SAVING ───────────────────────────
print("▸ Creating plots …")
plots = []

# longest 20 tracks   (frames vs x-coord)
dur = {tid: (end - st + 1)
       for tid, (st, end) in track_life.items() if st is not None}
top20 = sorted(dur.items(), key=lambda kv: -kv[1])[:20]

fig1 = plt.figure(figsize=(10, 6))
for tid, _ in top20:
    if tid in track_pos:
        fr, xs = zip(*track_pos[tid])
        plt.plot(fr, xs, label=f'ID {tid}')
plt.title('X vs Frame (Top-20 longest tracks)')
plt.xlabel('Frame')
plt.ylabel('X coordinate (px)')
plt.legend(ncol=2, fontsize='small')
plt.tight_layout()
plots.append(fig1)

# detections per frame
fig2 = plt.figure(figsize=(10, 4))
plt.plot(cluster_counts)
plt.title('Detections per slice')
plt.xlabel('Slice index')
plt.ylabel('Clusters')
plt.tight_layout()
plots.append(fig2)

# IoU quality histogram
all_iou = [v for lst in iou_scores.values() for v in lst]
if all_iou:
    fig3 = plt.figure(figsize=(8, 4))
    plt.hist(all_iou, bins=50, range=(0, 1))
    plt.title('IoU quality distribution')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plots.append(fig3)

# active tracks per slice
fig4 = plt.figure(figsize=(10, 4))
plt.plot(track_counts, 'r-', linewidth=2)
plt.title('Active tracks per slice')
plt.xlabel('Slice index')
plt.ylabel('Tracks')
plt.grid(alpha=0.3)
plt.tight_layout()
plots.append(fig4)

# slice length over time
fig5 = plt.figure(figsize=(10, 4))
plt.plot(slice_durations, 'g-', linewidth=2)
plt.title('Slice duration (velocity adaptive)')
plt.xlabel('Slice index')
plt.ylabel('Duration (ms)')
plt.grid(alpha=0.3)
plt.tight_layout()
plots.append(fig5)

# save everything
save_outputs(video_frames, plots)

print(f"▸ Finished.  Slices processed: {len(video_frames)}")
print(f"  Average Δt: {np.mean(slice_durations):.1f} ms  "
      f"(min {np.min(slice_durations):.1f} / max {np.max(slice_durations):.1f})")
