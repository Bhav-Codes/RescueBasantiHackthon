# src/image_to_graph.py
import math, json
from typing import Tuple, List, Dict
import numpy as np
import cv2

# Configurable color / speed mapping outside if needed
COLOR_SPEED = {"red":1.0, "blue":2.0, "green":3.0, "yellow":4.0}
COLOR_HSV_RANGES = {
    "red": [((0, 80, 50), (12, 255, 255)), ((160, 80, 50), (179, 255, 255))],
    "blue": [((90, 80, 50), (140, 255, 255))],
    "green": [((35, 60, 50), (90, 255, 255))],
    "yellow": [((15, 100, 100), (40, 255, 255))],
}

def load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def create_color_masks(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    masks = {}
    for color, ranges in COLOR_HSV_RANGES.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            mask_total = cv2.bitwise_or(mask_total, m)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)
        masks[color] = (mask_total > 0).astype(np.uint8)
    return masks

def skeletonize_fallback(bin_img: np.ndarray) -> np.ndarray:
    """
    Fast skeletonization:
      - Prefer cv2.ximgproc.thinning if available
      - Else prefer skimage.skeletonize if present
      - Else use iterative morphological skeleton (fast enough)
    Input: bin_img -> 0/1 uint8
    Output: 0/1 uint8 skeleton
    """
    if hasattr(cv2, "ximgproc"):
        try:
            thin = cv2.ximgproc.thinning((bin_img*255).astype(np.uint8),
                                        thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
            return (thin>0).astype(np.uint8)
        except Exception:
            pass
    try:
        from skimage.morphology import skeletonize as sk_skel
        thin = sk_skel(bin_img.astype(bool))
        return thin.astype(np.uint8)
    except Exception:
        pass
    # Morphological skeleton (fast)
    img = (bin_img*255).astype(np.uint8)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return (skel>0).astype(np.uint8)

def detect_black_circles(rgb: np.ndarray) -> List[Dict]:
    """Find large dark blobs (assumes black filled circles for start/target markers)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100: continue
        (x,y), r = cv2.minEnclosingCircle(cnt)
        circle_area = math.pi*(r**2)
        if circle_area <= 0: continue
        if area / circle_area > 0.4:
            circles.append({"x": float(x), "y": float(y), "r": float(r), "area": area})
    circles.sort(key=lambda c: c["x"])
    return circles

# def extract_graph_from_skeleton(skel: np.ndarray, color_masks: Dict[str, np.ndarray],
#                                 scale_pixels_per_unit: float = 50.0):
#     """
#     Returns:
#       - nodes: [ {"id":"N1","x":cx,"y":cy}, ... ]
#       - edges: [ {"from":"N1", "to":"N2", "color":"red", "length_units":..., "time":...}, ... ]
#     """
#     H,W = skel.shape
#     coords = np.column_stack(np.where(skel>0))  # (y,x)
#     if len(coords)==0:
#         return [], []
#     coords_xy = [(int(x),int(y)) for y,x in coords]
#     idx_map = -1*np.ones_like(skel, dtype=int)
#     for i,(x,y) in enumerate(coords_xy):
#         idx_map[y,x] = i
#     neighbors = {}
#     for i,(x,y) in enumerate(coords_xy):
#         neigh = []
#         for dy in [-1,0,1]:
#             for dx in [-1,0,1]:
#                 if dx==0 and dy==0: continue
#                 nxp, nyp = x+dx, y+dy
#                 if 0<=nxp<W and 0<=nyp<H and skel[nyp,nxp]:
#                     neigh.append(idx_map[nyp,nxp])
#         neighbors[i] = neigh
#     # Node pixels = degree != 2
#     node_pixels = [i for i,nbs in neighbors.items() if len(nbs) != 2]
#     node_mask = np.zeros_like(skel, dtype=np.uint8)
#     for i in node_pixels:
#         x,y = coords_xy[i]; node_mask[y,x] = 1
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
#     dil = cv2.dilate(node_mask, kernel)
#     num_labels, labels = cv2.connectedComponents(dil.astype(np.uint8))
#     nodes = []
#     for lab in range(1, num_labels):
#         ys, xs = np.where(labels==lab)
#         if len(xs)==0: continue
#         cx = float(xs.mean()); cy = float(ys.mean())
#         nodes.append({"id": f"N{lab}", "x": cx, "y": cy})
#     pixel_to_node = {}
#     for idx,(x,y) in enumerate(coords_xy):
#         lab = labels[y,x]
#         if lab>0:
#             pixel_to_node[idx] = f"N{lab}"
#     # Walk skeleton chains from node pixels to build edges
#     edges_tmp = []
#     visited = set()
#     for idx in range(len(coords_xy)):
#         if idx in visited: continue
#         if idx not in pixel_to_node: continue
#         u = pixel_to_node[idx]
#         for nb in neighbors[idx]:
#             if nb in pixel_to_node:
#                 v = pixel_to_node[nb]
#                 if u!=v:
#                     edges_tmp.append({"u":u,"v":v,"pts":[coords_xy[idx], coords_xy[nb]]})
#                 continue
#             path = [idx, nb]
#             prev = idx; cur = nb
#             while True:
#                 visited.add(cur)
#                 nbs = [n for n in neighbors[cur] if n != prev]
#                 if len(nbs)==0:
#                     break
#                 next_idx = nbs[0]
#                 path.append(next_idx)
#                 prev, cur = cur, next_idx
#                 if cur in pixel_to_node:
#                     break
#             if cur in pixel_to_node:
#                 v = pixel_to_node[cur]
#                 if u==v: continue
#                 pts = [coords_xy[p] for p in path]
#                 edges_tmp.append({"u":u,"v":v,"pts":pts})
#     # Deduplicate edges & compute lengths/colors
#     ded = {}
#     for e in edges_tmp:
#         key = tuple(sorted([e["u"], e["v"]]))
#         if key in ded:
#             # keep shorter
#             prevlen = sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in zip(ded[key]["pts"][:-1], ded[key]["pts"][1:]))
#             curlen = sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in zip(e["pts"][:-1], e["pts"][1:]))
#             if curlen < prevlen:
#                 ded[key] = e
#         else:
#             ded[key] = e
#     final_edges = []
#     for (u,v), e in ded.items():
#         pts = e["pts"]
#         length_px = sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1,p2 in zip(pts[:-1], pts[1:]))
#         # color counts
#         color_counts = {c:0 for c in COLOR_HSV_RANGES.keys()}
#         for (x,y) in pts:
#             for yy in range(max(0,y-1), min(H,y+2)):
#                 for xx in range(max(0,x-1), min(W,x+2)):
#                     for c,m in color_masks.items():
#                         if m[yy,xx]:
#                             color_counts[c] += 1
#         dominant = max(color_counts.items(), key=lambda kv: kv[1])[0] if any(v>0 for v in color_counts.values()) else "red"
#         length_units = length_px / scale_pixels_per_unit
#         time_sec = length_units / COLOR_SPEED.get(dominant, 1.0)
#         final_edges.append({
#             "from": u, "to": v,
#             "color": dominant,
#             "length_units": length_units,
#             "time": time_sec,
#             "length_px": length_px
#         })
#     return nodes, final_edges

def extract_graph_from_skeleton(skel: np.ndarray, color_masks: Dict[str, np.ndarray],
                                scale_pixels_per_unit: float = 50.0):
    """
    Returns:
      - nodes: [ {"id":"N1","x":cx,"y":cy}, ... ]
      - edges: [ {"from":"N1", "to":"N2", "color":"red", "length_units":..., "time":...}, ... ]

    Now detects:
      - Intersections (degree != 2)
      - Turns (degree == 2 but neighbors not collinear)
    """
    H, W = skel.shape
    coords = np.column_stack(np.where(skel > 0))  # (y,x)
    if len(coords) == 0:
        return [], []

    coords_xy = [(int(x), int(y)) for y, x in coords]
    idx_map = -1 * np.ones_like(skel, dtype=int)
    for i, (x, y) in enumerate(coords_xy):
        idx_map[y, x] = i

    # Build neighbors dict
    neighbors = {}
    for i, (x, y) in enumerate(coords_xy):
        neigh = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nxp, nyp = x + dx, y + dy
                if 0 <= nxp < W and 0 <= nyp < H and skel[nyp, nxp]:
                    neigh.append(idx_map[nyp, nxp])
        neighbors[i] = neigh

    # Detect node pixels: intersections OR turns
    node_pixels = []
    for i, nbs in neighbors.items():
        if len(nbs) != 2:
            node_pixels.append(i)
        elif len(nbs) == 2:
            # check if turn (non-collinear)
            x0, y0 = coords_xy[i]
            n1, n2 = nbs
            x1, y1 = coords_xy[n1]
            x2, y2 = coords_xy[n2]
            # vectors
            v1 = (x1 - x0, y1 - y0)
            v2 = (x2 - x0, y2 - y0)
            # cross product to check non-collinear
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if abs(cross) > 0:
                node_pixels.append(i)

    # Dilate nodes and label
    node_mask = np.zeros_like(skel, dtype=np.uint8)
    for i in node_pixels:
        x, y = coords_xy[i]
        node_mask[y, x] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dil = cv2.dilate(node_mask, kernel)
    num_labels, labels = cv2.connectedComponents(dil.astype(np.uint8))

    nodes = []
    for lab in range(1, num_labels):
        ys, xs = np.where(labels == lab)
        if len(xs) == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        nodes.append({"id": f"N{lab}", "x": cx, "y": cy})

    # Map skeleton pixels to nearest node
    pixel_to_node = {}
    for idx, (x, y) in enumerate(coords_xy):
        lab = labels[y, x]
        if lab > 0:
            pixel_to_node[idx] = f"N{lab}"

    # Walk skeleton chains to build edges
    edges_tmp = []
    visited = set()
    for idx in range(len(coords_xy)):
        if idx in visited or idx not in pixel_to_node:
            continue
        u = pixel_to_node[idx]
        for nb in neighbors[idx]:
            if nb in pixel_to_node:
                v = pixel_to_node[nb]
                if u != v:
                    edges_tmp.append({"u": u, "v": v, "pts": [coords_xy[idx], coords_xy[nb]]})
                continue
            path = [idx, nb]
            prev, cur = idx, nb
            while True:
                visited.add(cur)
                nbs = [n for n in neighbors[cur] if n != prev]
                if len(nbs) == 0:
                    break
                next_idx = nbs[0]
                path.append(next_idx)
                prev, cur = cur, next_idx
                if cur in pixel_to_node:
                    break
            if cur in pixel_to_node:
                v = pixel_to_node[cur]
                if u == v:
                    continue
                pts = [coords_xy[p] for p in path]
                edges_tmp.append({"u": u, "v": v, "pts": pts})

    # Deduplicate edges & compute lengths/colors
    ded = {}
    for e in edges_tmp:
        key = tuple(sorted([e["u"], e["v"]]))
        if key in ded:
            # keep shorter
            prevlen = sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                          for p1, p2 in zip(ded[key]["pts"][:-1], ded[key]["pts"][1:]))
            curlen = sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                         for p1, p2 in zip(e["pts"][:-1], e["pts"][1:]))
            if curlen < prevlen:
                ded[key] = e
        else:
            ded[key] = e

    final_edges = []
    for (u, v), e in ded.items():
        pts = e["pts"]
        length_px = sum(math.hypot(p2[0]-p1[0], p2[1]-p1[1]) for p1, p2 in zip(pts[:-1], pts[1:]))
        # color counts
        color_counts = {c: 0 for c in COLOR_HSV_RANGES.keys()}
        for (x, y) in pts:
            for yy in range(max(0, y-1), min(H, y+2)):
                for xx in range(max(0, x-1), min(W, x+2)):
                    for c, m in color_masks.items():
                        if m[yy, xx]:
                            color_counts[c] += 1
        dominant = max(color_counts.items(), key=lambda kv: kv[1])[0] \
            if any(v > 0 for v in color_counts.values()) else "red"
        length_units = length_px / scale_pixels_per_unit
        time_sec = length_units / COLOR_SPEED.get(dominant, 1.0)
        # final_edges.append({
        #     "from": u, "to": v,
        #     "color": dominant,
        #     "length_units": length_units,
        #     "time": time_sec,
        #     "length_px": length_px
        # })
        final_edges.append({
            "from": u, "to": v,
            "color": dominant,
            "length_units": length_units,
            "time": time_sec,
            "length_px": length_px,
            "pts": pts   # keep this!
        })

    return nodes, final_edges