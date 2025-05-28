# #!/usr/bin/env python3
# """
# Nearest-neighbour corner tracker

# Author: <Salim>
# """

# import os, cv2, math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist

# # ───────── Settings (identical to IoU script) ──────────────────────
# IMAGE_DIR        = '/home/salblooshi/Desktop/Master/Term2/Preception/Project2/shapes_6dof/images'
# EVENTS_FILE      = '/home/salblooshi/Desktop/Master/Term2/Preception/Project2/corner_workspace/src/parsed_feature_events.txt'

# FPS              = 22.68
# SKIP_IMAGES      = 150
# TIME_OFFSET_MS   = 0

# SPATIAL_RADIUS   = 4
# TEMPORAL_WINDOW  = 5
# DENSITY_THRESH   = 6
# MIN_CLUSTER_SIZE = 5
# MAX_CLUSTERS     = 200

# TRAIL_LENGTH     = 15
# MAX_MISSED       = 2

# TOTAL_FRAMES     = 1356      

# # ───────── Load & filter events ────────────────────────────────────
# df = pd.read_csv(EVENTS_FILE, header=None,
#                  names=['x', 'y', 'ts', 'polarity'])
# df['ts'] = (df['ts'] - df['ts'].min()) / 1e6          # ns → ms
# df       = df[df['ts'] >= TIME_OFFSET_MS]

# def spatiotemporal_filter(df, r, t_win, thresh):
#     if df.empty:
#         return df.copy()
#     xyz = df[['x','y','ts']].to_numpy()
#     xyz[:,2] /= t_win
#     nbrs = NearestNeighbors(radius=r).fit(xyz)
#     dists, _ = nbrs.radius_neighbors(xyz)
#     keep = np.array([len(d)-1 for d in dists]) >= thresh
#     return df[keep]

# df = spatiotemporal_filter(df, SPATIAL_RADIUS,
#                            TEMPORAL_WINDOW, DENSITY_THRESH)

# WINDOW_MS = 1000.0 / FPS
# df['frame'] = ((df['ts'] - TIME_OFFSET_MS) // WINDOW_MS).astype(int)
# if SKIP_IMAGES:
#     df = df[df['frame'] >= SKIP_IMAGES]
#     df['frame'] -= SKIP_IMAGES

# events_by_frame = df.groupby('frame')[['x','y']]

# # ───────── Image list ──────────────────────────────────────────────
# imgs = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith('.png'))
# imgs = imgs[SKIP_IMAGES:]
# if not imgs:
#     raise RuntimeError("No images found after skipping!")

# # ───────── Tracker state ───────────────────────────────────────────
# next_id           = 0
# active_tracks     = {}                 # tid → (x, y)
# missed_counts     = defaultdict(int)

# trails            = defaultdict(list)  # short live trail
# full_trails       = defaultdict(list)  # NEVER pruned

# # Stats
# track_life              = {}                  # tid → [first, last]
# track_pos               = defaultdict(list)   # tid → [(frame,x)]
# det_per_frame           = []
# track_count_per_frame   = []

# # ───────── Display window ──────────────────────────────────────────
# cv2.namedWindow('Corners+Tracks', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Corners+Tracks', 800, 600)

# # ───────── Main loop ───────────────────────────────────────────────
# for frame_idx, fname in enumerate(imgs):
#     t_now = frame_idx * WINDOW_MS
#     img   = cv2.imread(os.path.join(IMAGE_DIR, fname))
#     if img is None:
#         continue

#     # 1  Events → points
#     pts = (events_by_frame.get_group(frame_idx)[['x','y']].to_numpy()
#            if frame_idx in events_by_frame.groups else
#            np.empty((0,2)))

#     # 2  K-Means clustering
#     centroids = []
#     if len(pts) >= MIN_CLUSTER_SIZE:
#         k = min(max(1, int(np.sqrt(len(pts)))), MAX_CLUSTERS)
#         labels = KMeans(n_clusters=k, random_state=frame_idx,
#                         n_init=10).fit(pts).labels_
#         for cid in np.unique(labels):
#             cpts = pts[labels == cid]
#             if len(cpts) >= MIN_CLUSTER_SIZE:
#                 centroids.append(cpts.mean(axis=0))

#     det_per_frame.append(len(centroids))

#     # 3  Nearest-neighbour association
#     unmatched  = set(range(len(centroids)))
#     matched_id = set()
#     if centroids and active_tracks:
#         new_pos  = np.array(centroids)
#         tids     = list(active_tracks.keys())
#         prev_pos = np.array([active_tracks[tid] for tid in tids])
#         dmat     = cdist(new_pos, prev_pos)

#         for i, row in enumerate(dmat):
#             j   = np.argmin(row)
#             tid = tids[j]

#             # dynamic threshold
#             if len(trails[tid]) >= 2:
#                 vx = trails[tid][-1][0] - trails[tid][-2][0]
#                 vy = trails[tid][-1][1] - trails[tid][-2][1]
#                 speed = np.hypot(vx, vy)
#             else:
#                 speed = 0.0
#             thr = max(6, min(30, speed*1.5))

#             if row[j] < thr:
#                 active_tracks[tid] = centroids[i]
#                 trails[tid].append(tuple(map(int, centroids[i])))
#                 trails[tid] = trails[tid][-TRAIL_LENGTH:]          # prune
#                 full_trails[tid].append(tuple(map(int, centroids[i])))

#                 missed_counts[tid] = 0
#                 matched_id.add(tid)
#                 unmatched.discard(i)

#                 track_pos[tid].append((frame_idx, centroids[i][0]))
#                 track_life.setdefault(tid, [frame_idx, frame_idx])[1] = frame_idx

#     # 4  Spawn new tracks
#     for i in unmatched:
#         active_tracks[next_id] = centroids[i]
#         trails[next_id]        = [tuple(map(int, centroids[i]))]
#         full_trails[next_id]   = [tuple(map(int, centroids[i]))]
#         missed_counts[next_id] = 0

#         track_pos[next_id].append((frame_idx, centroids[i][0]))
#         track_life[next_id]    = [frame_idx, frame_idx]
#         next_id += 1

#     # 5  Age / prune
#     for tid in list(active_tracks):
#         if tid not in matched_id:
#             missed_counts[tid] += 1
#         if missed_counts[tid] > MAX_MISSED:
#             del active_tracks[tid]
#             del trails[tid]           # keep full_trails intact
#             del missed_counts[tid]

#     track_count_per_frame.append(len(active_tracks))

#     # 6  Draw
#     for (x, y) in centroids:
#         cv2.circle(img, (int(x), int(y)), 2, (0,165,0), -1)

#     for tid, pos in active_tracks.items():
#         x, y = map(int, pos)
#         cv2.putText(img, str(tid), (x, y-4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
#         for i in range(1, len(trails[tid])):
#             cv2.line(img, trails[tid][i-1], trails[tid][i],
#                      (0,255,255), 1)

#     cv2.imshow('Corners+Tracks',
#                cv2.resize(img, None, fx=3, fy=3,
#                           interpolation=cv2.INTER_NEAREST))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.waitKey(0); cv2.destroyAllWindows()

# # ───────── Statistics & plots ──────────────────────────────────────
# durations = {tid:(end-start+1) for tid,(start,end) in track_life.items()}
# top20     = sorted(durations.items(), key=lambda x:-x[1])[:20]
# x_full    = np.arange(TOTAL_FRAMES)

# # 1  X vs Frame (Top-20)
# plt.figure(figsize=(10,6))
# for tid,_ in top20:
#     if tid in track_pos:
#         fr, xs = zip(*track_pos[tid])
#         plt.plot(fr, xs, label=f'ID {tid}')
# plt.title('X vs Frame (Top20)')
# plt.xlabel('Frame'); plt.ylabel('X')
# plt.xlim(0, TOTAL_FRAMES-1)
# plt.legend(ncol=2, fontsize='small')
# plt.tight_layout()

# # 2  Detections / frame
# plt.figure(figsize=(10,4))
# plt.plot(x_full[:len(det_per_frame)], det_per_frame)
# plt.title('Detections/frame')
# plt.xlabel('Frame'); plt.ylabel('Count')
# plt.xlim(0, TOTAL_FRAMES-1); plt.ylim(0, 200)
# plt.tight_layout()

# # 3  Track lifespans
# plt.figure(figsize=(8,4))
# plt.hist(list(durations.values()), bins=30, range=(0, TOTAL_FRAMES))
# plt.title('Track Lifespans')
# plt.xlabel('Frames'); plt.ylabel('Count')
# plt.tight_layout()

# # 4  Active tracks over time
# plt.figure(figsize=(10,4))
# plt.plot(x_full[:len(track_count_per_frame)], track_count_per_frame)
# plt.title('Active Tracks Over Time')
# plt.xlabel('Frame'); plt.ylabel('# Active Tracks')
# plt.xlim(0, TOTAL_FRAMES-1)
# plt.tight_layout()

# # 5  Average track speeds (ALL tracks)
# avg_speeds = []
# for tr in full_trails.values():
#     if len(tr) > 1:
#         d = [np.hypot(tr[i][0] - tr[i-1][0],
#                       tr[i][1] - tr[i-1][1]) for i in range(1, len(tr))]
#         avg_speeds.append(np.mean(d))
# plt.figure(figsize=(8,4))
# plt.hist(avg_speeds, bins=np.linspace(0, 20, 41))   # 0.5-pixel bins
# plt.title('Average Track Speeds')
# plt.xlabel('Pixels/frame'); plt.ylabel('Track Count')
# plt.tight_layout()

# plt.show()



#!/usr/bin/env python3
"""
Nearest-neighbour corner tracker

Author: <Salim>
"""

import os, cv2, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# ───────── Settings (identical to IoU script) ──────────────────────
IMAGE_DIR        = '/home/salblooshi/Desktop/Master/Term2/Preception/Project2/shapes_6dof/images'
EVENTS_FILE      = '/home/salblooshi/Desktop/Master/Term2/Preception/Project2/corner_workspace/src/parsed_feature_events.txt'


FPS              = 22.68
SKIP_IMAGES      = 0
TIME_OFFSET_MS   = 0

SPATIAL_RADIUS   = 4
TEMPORAL_WINDOW  = 5
DENSITY_THRESH   = 2
MIN_CLUSTER_SIZE = 5
MAX_CLUSTERS     = 1000

TRAIL_LENGTH     = 15
MAX_MISSED       = 2

TOTAL_FRAMES     = 1298          # force plot range

# ───────── Save output directory ────────────────────────────────────
def save_outputs(video_frames, plots_to_save, output_dir='plots_videos'):
    """
    Save video output and plots to a specified directory.
    
    Args:
        video_frames: List of frames (numpy arrays) to save as video
        plots_to_save: List of matplotlib figure objects to save
        output_dir: Directory name to save outputs (default: 'shapes_6dof')
    """
    # Create output directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, output_dir)
    os.makedirs(save_path, exist_ok=True)
    
    # Save video if frames are provided
    if video_frames:
        video_path = os.path.join(save_path, 'nn_tracking_output.avi')
        height, width = video_frames[0].shape[:2]
        
        # Try different codecs
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
        
        if not out.isOpened():
            print("XVID codec not available, trying MJPG...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
        
        for frame in video_frames:
            out.write(frame)
        
        out.release()
        print(f"Video saved to: {video_path}")
        
        # Also save as high-quality image sequence
        img_seq_path = os.path.join(save_path, 'nn_frames')
        os.makedirs(img_seq_path, exist_ok=True)
        print(f"Saving image sequence to: {img_seq_path}")
        
        for i, frame in enumerate(video_frames):
            frame_path = os.path.join(img_seq_path, f'frame_{i:04d}.png')
            cv2.imwrite(frame_path, frame)
        
        print(f"Saved {len(video_frames)} frames as PNG sequence")
    
    # Save plots
    plot_names = ['nn_x_vs_frame_top20.png', 'nn_detections_per_frame.png', 
                  'nn_track_lifespans.png', 'nn_active_tracks_over_time.png', 
                  'nn_average_track_speeds.png']
    
    for fig, name in zip(plots_to_save, plot_names):
        plot_path = os.path.join(save_path, name)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")

# ───────── Load & filter events ────────────────────────────────────
df = pd.read_csv(EVENTS_FILE, header=None,
                 names=['x', 'y', 'ts', 'polarity'])
df['ts'] = (df['ts'] - df['ts'].min()) / 1e6          # ns → ms
df       = df[df['ts'] >= TIME_OFFSET_MS]

def spatiotemporal_filter(df, r, t_win, thresh):
    if df.empty:
        return df.copy()
    xyz = df[['x','y','ts']].to_numpy()
    xyz[:,2] /= t_win
    nbrs = NearestNeighbors(radius=r).fit(xyz)
    dists, _ = nbrs.radius_neighbors(xyz)
    keep = np.array([len(d)-1 for d in dists]) >= thresh
    return df[keep]

df = spatiotemporal_filter(df, SPATIAL_RADIUS,
                           TEMPORAL_WINDOW, DENSITY_THRESH)

WINDOW_MS = 1000.0 / FPS
df['frame'] = ((df['ts'] - TIME_OFFSET_MS) // WINDOW_MS).astype(int)
if SKIP_IMAGES:
    df = df[df['frame'] >= SKIP_IMAGES]
    df['frame'] -= SKIP_IMAGES

events_by_frame = df.groupby('frame')[['x','y']]

# ───────── Image list ──────────────────────────────────────────────
imgs = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith('.png'))
imgs = imgs[SKIP_IMAGES:]
if not imgs:
    raise RuntimeError("No images found after skipping!")

# ───────── Tracker state ───────────────────────────────────────────
next_id           = 0
active_tracks     = {}                 # tid → (x, y)
missed_counts     = defaultdict(int)

trails            = defaultdict(list)  # short live trail
full_trails       = defaultdict(list)  # NEVER pruned

# Stats
track_life              = {}                  # tid → [first, last]
track_pos               = defaultdict(list)   # tid → [(frame,x)]
det_per_frame           = []
track_count_per_frame   = []

# List to store video frames
video_frames = []

# ───────── Display window ──────────────────────────────────────────
cv2.namedWindow('Corners+Tracks', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Corners+Tracks', 800, 600)

# ───────── Main loop ───────────────────────────────────────────────
for frame_idx, fname in enumerate(imgs):
    t_now = frame_idx * WINDOW_MS
    img   = cv2.imread(os.path.join(IMAGE_DIR, fname))
    if img is None:
        continue

    # Make a copy for saving
    img_save = img.copy()

    # 1  Events → points
    pts = (events_by_frame.get_group(frame_idx)[['x','y']].to_numpy()
           if frame_idx in events_by_frame.groups else
           np.empty((0,2)))

    # 2  K-Means clustering
    centroids = []
    if len(pts) >= MIN_CLUSTER_SIZE:
        k = min(max(1, int(np.sqrt(len(pts)))), MAX_CLUSTERS)
        labels = KMeans(n_clusters=k, random_state=frame_idx,
                        n_init=10).fit(pts).labels_
        for cid in np.unique(labels):
            cpts = pts[labels == cid]
            if len(cpts) >= MIN_CLUSTER_SIZE:
                centroids.append(cpts.mean(axis=0))

    det_per_frame.append(len(centroids))

    # 3  Nearest-neighbour association
    unmatched  = set(range(len(centroids)))
    matched_id = set()
    if centroids and active_tracks:
        new_pos  = np.array(centroids)
        tids     = list(active_tracks.keys())
        prev_pos = np.array([active_tracks[tid] for tid in tids])
        dmat     = cdist(new_pos, prev_pos)

        for i, row in enumerate(dmat):
            j   = np.argmin(row)
            tid = tids[j]

            # dynamic threshold
            if len(trails[tid]) >= 2:
                vx = trails[tid][-1][0] - trails[tid][-2][0]
                vy = trails[tid][-1][1] - trails[tid][-2][1]
                speed = np.hypot(vx, vy)
            else:
                speed = 0.0
            thr = max(6, min(30, speed*1.5))

            if row[j] < thr:
                active_tracks[tid] = centroids[i]
                trails[tid].append(tuple(map(int, centroids[i])))
                trails[tid] = trails[tid][-TRAIL_LENGTH:]          # prune
                full_trails[tid].append(tuple(map(int, centroids[i])))

                missed_counts[tid] = 0
                matched_id.add(tid)
                unmatched.discard(i)

                track_pos[tid].append((frame_idx, centroids[i][0]))
                track_life.setdefault(tid, [frame_idx, frame_idx])[1] = frame_idx

    # 4  Spawn new tracks
    for i in unmatched:
        active_tracks[next_id] = centroids[i]
        trails[next_id]        = [tuple(map(int, centroids[i]))]
        full_trails[next_id]   = [tuple(map(int, centroids[i]))]
        missed_counts[next_id] = 0

        track_pos[next_id].append((frame_idx, centroids[i][0]))
        track_life[next_id]    = [frame_idx, frame_idx]
        next_id += 1

    # 5  Age / prune
    for tid in list(active_tracks):
        if tid not in matched_id:
            missed_counts[tid] += 1
        if missed_counts[tid] > MAX_MISSED:
            del active_tracks[tid]
            del trails[tid]           # keep full_trails intact
            del missed_counts[tid]

    track_count_per_frame.append(len(active_tracks))

    # 6  Draw on both display and save images
    for (x, y) in centroids:
        cv2.circle(img, (int(x), int(y)), 2, (0,165,0), -1)
        cv2.circle(img_save, (int(x), int(y)), 2, (0,165,0), -1)

    for tid, pos in active_tracks.items():
        x, y = map(int, pos)
        cv2.putText(img, str(tid), (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        cv2.putText(img_save, str(tid), (x, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
        
        for i in range(1, len(trails[tid])):
            cv2.line(img, trails[tid][i-1], trails[tid][i],
                     (0,255,255), 1)
            cv2.line(img_save, trails[tid][i-1], trails[tid][i],
                     (0,255,255), 1)

    # Save frame for video
    video_frames.append(img_save)

    cv2.imshow('Corners+Tracks',
               cv2.resize(img, None, fx=3, fy=3,
                          interpolation=cv2.INTER_NEAREST))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# ───────── Statistics & plots ──────────────────────────────────────
durations = {tid:(end-start+1) for tid,(start,end) in track_life.items()}
top20     = sorted(durations.items(), key=lambda x:-x[1])[:20]
x_full    = np.arange(TOTAL_FRAMES)

# List to store figure objects
plots_to_save = []

# 1  X vs Frame (Top-20)
fig1 = plt.figure(figsize=(10,6))
for tid,_ in top20:
    if tid in track_pos:
        fr, xs = zip(*track_pos[tid])
        plt.plot(fr, xs, label=f'ID {tid}')
plt.title('X vs Frame (Top20)')
plt.xlabel('Frame'); plt.ylabel('X')
plt.xlim(0, TOTAL_FRAMES-1)
plt.legend(ncol=2, fontsize='small')
plt.tight_layout()
plots_to_save.append(fig1)

# 2  Detections / frame
fig2 = plt.figure(figsize=(10,4))
plt.plot(x_full[:len(det_per_frame)], det_per_frame)
plt.title('Detections/frame')
plt.xlabel('Frame'); plt.ylabel('Count')
plt.xlim(0, TOTAL_FRAMES-1); plt.ylim(0, 200)
plt.tight_layout()
plots_to_save.append(fig2)

# 3  Track lifespans
fig3 = plt.figure(figsize=(8,4))
plt.hist(list(durations.values()), bins=30, range=(0, TOTAL_FRAMES))
plt.title('Track Lifespans')
plt.xlabel('Frames'); plt.ylabel('Count')
plt.tight_layout()
plots_to_save.append(fig3)

# 4  Active tracks over time
fig4 = plt.figure(figsize=(10,4))
plt.plot(x_full[:len(track_count_per_frame)], track_count_per_frame)
plt.title('Active Tracks Over Time')
plt.xlabel('Frame'); plt.ylabel('# Active Tracks')
plt.xlim(0, TOTAL_FRAMES-1)
plt.tight_layout()
plots_to_save.append(fig4)

# 5  Average track speeds (ALL tracks)
avg_speeds = []
for tr in full_trails.values():
    if len(tr) > 1:
        d = [np.hypot(tr[i][0] - tr[i-1][0],
                      tr[i][1] - tr[i-1][1]) for i in range(1, len(tr))]
        avg_speeds.append(np.mean(d))
fig5 = plt.figure(figsize=(8,4))
plt.hist(avg_speeds, bins=np.linspace(0, 20, 41))   # 0.5-pixel bins
plt.title('Average Track Speeds')
plt.xlabel('Pixels/frame'); plt.ylabel('Track Count')
plt.tight_layout()
plots_to_save.append(fig5)
# 4. Track Lifespan Histogram (how many IDs lived for X frames)
fig4 = plt.figure(figsize=(10,4))
lifespans = list(durations.values())  # list of frame counts per ID
plt.hist(lifespans, bins=range(1, max(lifespans)+2), edgecolor='black', align='left')
plt.title('Histogram: Track Lifespans')
plt.xlabel('Frames Survived')
plt.ylabel('Number of Track IDs')
plt.tight_layout()
plots_to_save.append(fig4)


# Save all outputs
save_outputs(video_frames, plots_to_save)

plt.show()
