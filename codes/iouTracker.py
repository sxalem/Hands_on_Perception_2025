# #!/usr/bin/env python3

# Author: <Salim>
# """

# import os, cv2, math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans
# from tracker import iou_tracker                  # your wrapper

# # ───────── Parameters ───────────────────────────────────────────────
# IMAGE_DIR        = '/home/salblooshi/Desktop/HOP/iou_kalman_tracker/boxes_6dof/images'
# EVENTS_FILE      = '/home/salblooshi/Desktop/HOP/iou_kalman_tracker/corner_workspace/src/parsed_feature_events_box6dof.txt'

# FPS              = 21.74   #22.68 for shpes_translation and shapes_6dof
# SKIP_IMAGES      = 0
# TIME_OFFSET_MS   = 0

# SPATIAL_RADIUS   = 4
# TEMPORAL_WINDOW  = 5
# DENSITY_THRESH   = 2
# MIN_CLUSTER_SIZE = 5
# MAX_CLUSTERS     = 1000

# SIGMA_IOU        = 0.20        # IoU match threshold
# T_MIN            = 60          # min consecutive frames to confirm a track
# MAX_MISSED       = 2           # frames a confirmed track may miss
# TRAIL_LENGTH     = 15          # positions kept for trail drawing

# SCALE_FACTOR     = 10.0        # velocity → box size scale
# BOX_MIN_SIZE     = 10
# BOX_MAX_SIZE     = 30
# BOX_SIZE_DET     = 30           # fixed detection box size

# WINDOW_MS        = 1000.0 / FPS

# # ───────── Helper functions ─────────────────────────────────────────
# def bbox_iou(b1, b2):
#     xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
#     xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
#     inter  = max(0, xB-xA) * max(0, yB-yA)
#     if inter == 0:
#         return 0.0
#     a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
#     a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
#     return inter / (a1 + a2 - inter)

# def spatiotemporal_filter(df, r, t_win, thresh):
#     if df.empty:
#         return df.copy()
#     pts = df[['x','y','ts']].to_numpy()
#     pts[:,2] /= t_win
#     nbrs = NearestNeighbors(radius=r).fit(pts)
#     dists, _ = nbrs.radius_neighbors(pts)
#     keep = np.array([len(d)-1 for d in dists]) >= thresh
#     return df[keep]

# # ───────── Load events ──────────────────────────────────────────────
# df = pd.read_csv(EVENTS_FILE, header=None,
#                  names=['x','y','ts','polarity'])
# df['ts'] = (df['ts'] - df['ts'].min()) / 1e6         # ns → ms
# df       = df[df['ts'] >= TIME_OFFSET_MS]
# df['frame'] = ((df['ts'] - TIME_OFFSET_MS) // WINDOW_MS).astype(int)
# if SKIP_IMAGES:
#     df = df[df['frame'] >= SKIP_IMAGES]
#     df['frame'] -= SKIP_IMAGES

# df = spatiotemporal_filter(df, SPATIAL_RADIUS,
#                            TEMPORAL_WINDOW, DENSITY_THRESH)
# events_by_frame = df.groupby('frame')[['x','y','ts']]

# all_imgs = sorted(f for f in os.listdir(IMAGE_DIR)
#                   if f.endswith('.png'))[SKIP_IMAGES:]
# if not all_imgs:
#     raise RuntimeError("No images found after skipping!")
# total_frames = len(all_imgs)

# # ───────── Tracker & bookkeeping ────────────────────────────────────
# tracker       = iou_tracker.VehicleTracker()
# missed_counts = {}
# trails        = defaultdict(list)

# det_per_frame = []                          # detections per frame
# track_pos     = defaultdict(list)           # {tid:[(frame,x), …]}
# track_life    = {}                          # {tid:[first,last]}
# iou_scores    = defaultdict(list)           # {tid:[iou, …]}

# # ───────── Display window ──────────────────────────────────────────
# cv2.namedWindow('Corners+Tracks', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Corners+Tracks', 800, 600)

# # ───────── Main loop ───────────────────────────────────────────────
# for frame_idx, fname in enumerate(all_imgs):
#     t_now = frame_idx * WINDOW_MS

#     # 1) get events for this frame
#     pts = (events_by_frame.get_group(frame_idx)[['x','y']].to_numpy()
#            if frame_idx in events_by_frame.groups else
#            np.empty((0,2)))

#     # 2) K-Means clustering → centroids
#     centroids = []
#     if len(pts) >= MIN_CLUSTER_SIZE:
#         k = min(max(1, int(np.sqrt(len(pts)))), MAX_CLUSTERS)
#         km = KMeans(n_clusters=k, random_state=frame_idx).fit(pts)
#         for cid in np.unique(km.labels_):
#             if np.sum(km.labels_ == cid) >= MIN_CLUSTER_SIZE:
#                 centroids.append(km.cluster_centers_[cid])

#     # 3) centroids → detection boxes
#     dets = np.array([(int(x-BOX_SIZE_DET/2), int(y-BOX_SIZE_DET/2),
#                       int(x+BOX_SIZE_DET/2), int(y+BOX_SIZE_DET/2))
#                      for x,y in centroids], dtype=np.int32)
#     det_per_frame.append(len(dets))

#     # 4) tracker update
#     prev_counts = {tr['id']: len(tr['bboxes']) for tr in tracker.Ta}
#     tracker.track_iou(dets, t_now, sigma_iou=SIGMA_IOU, t_min=T_MIN)

#     # IoU quality bookkeeping
#     if len(dets) and tracker.Ta:
#         for db in dets:
#             best_iou, best_tid = 0., None
#             for tr in tracker.Ta:
#                 tid = tr['id']
#                 x_pred, _ = tr['kf'].predict_data_association(t_now)
#                 cx, cy = x_pred.flatten()[:2]

#                 # Predicted box
#                 box = [int(cx - BOX_SIZE_DET/2), int(cy - BOX_SIZE_DET/2),
#                     int(cx + BOX_SIZE_DET/2), int(cy + BOX_SIZE_DET/2)]

#                 iou = bbox_iou(db, box)
#                 if iou > best_iou:
#                     best_iou, best_tid = iou, tid
#             if best_tid is not None:
#                 iou_scores[best_tid].append(best_iou)


#     # prune missed
#     new_Ta = []
#     for tr in tracker.Ta:
#         tid = tr['id']
#         missed_counts[tid] = 0 if len(tr['bboxes']) > prev_counts.get(tid,0) \
#                                 else missed_counts.get(tid,0) + 1
#         if missed_counts[tid] <= MAX_MISSED:
#             new_Ta.append(tr)
#         else:
#             missed_counts.pop(tid, None)
#     tracker.Ta = new_Ta

#     # life / position stats
#     for tr in tracker.Ta:
#         tid       = tr['id']
#         x_pred,_  = tr['kf'].predict_data_association(t_now)
#         cx, cy    = x_pred.flatten()[:2]

#         if tid not in track_life:
#             track_life[tid] = [frame_idx, frame_idx]
#         else:
#             track_life[tid][1] = frame_idx
#         track_pos[tid].append((frame_idx, cx))

#     # 5) visualisation
#     img = cv2.imread(os.path.join(IMAGE_DIR, fname))
#     if img is None: continue

#     for x,y in centroids:
#         cv2.circle(img, (int(x), int(y)), 2, (0,165,0), -1)

#     for tr in tracker.Ta:
#         tid       = tr['id']
#         x_pred,_  = tr['kf'].predict_data_association(t_now)
#         cx, cy    = x_pred.flatten()[:2]

#         last = trails[tid][-1] if trails[tid] else (cx, cy)
#         speed = math.hypot(cx-last[0], cy-last[1])
#         box   = min(max(int(speed*SCALE_FACTOR), BOX_MIN_SIZE), BOX_MAX_SIZE)

#         x0,y0 = int(cx-box/2), int(cy-box/2)
#         x1,y1 = int(cx+box/2), int(cy+box/2)
#         cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),1)
#         cv2.putText(img,str(tid),(x0,y0-4),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)

#         trails[tid].append((int(cx),int(cy)))
#         if len(trails[tid]) > TRAIL_LENGTH:
#             trails[tid] = trails[tid][-TRAIL_LENGTH:]
#         for i in range(1,len(trails[tid])):
#             cv2.line(img,trails[tid][i-1],trails[tid][i],
#                      (0,255,255),1)

#     img_large = cv2.resize(img,None,fx=3.0,fy=3.0,
#                            interpolation=cv2.INTER_NEAREST)
#     cv2.imshow('Corners+Tracks',img_large)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # ────────── Stats & plots ───────────────────────────────────────────

# # 4. IoU Quality
# all_ious = [iou for lst in iou_scores.values() for iou in lst]
# plt.figure(figsize=(8,4))
# plt.hist(all_ious, bins=50, range=(0,1))  # ← higher resolution
# plt.title('IoU Quality')
# plt.xlabel('IoU')
# plt.ylabel('Count')
# plt.tight_layout()

# plt.show()

# 4. Track Lifespan Histogram (how many IDs lived for X frames)
# fig4 = plt.figure(figsize=(10,4))
# lifespans = list(durations.values())  # list of frame counts per ID
# plt.hist(lifespans, bins=range(1, max(lifespans)+2), edgecolor='black', align='left')
# plt.title('Histogram: Track Lifespans')
# plt.xlabel('Frames Survived')
# plt.ylabel('Number of Track IDs')
# plt.tight_layout()
# plots_to_save.append(fig4)
# # Save all outputs
# save_outputs(video_frames, plots_to_save)

# plt.show()




































