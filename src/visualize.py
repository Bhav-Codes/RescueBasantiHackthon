# # src/visualize.py
# from PIL import Image, ImageDraw
# from typing import List, Dict

# def draw_overlay(orig_rgb, color_masks, skeleton, nodes, edges, path_nodes, out_path):
#     """
#     orig_rgb: numpy RGB image
#     color_masks: dict[color] -> binary mask
#     skeleton: binary image (unused directly here but kept for reference)
#     nodes: [{"id","x","y"}, ...]
#     edges: [{"from","to","color", ...}, ...]
#     path_nodes: ['N1','N2', ...]
#     Saves PNG to out_path.
#     """
#     import numpy as np
#     H,W,_ = orig_rgb.shape
#     img = Image.fromarray(orig_rgb.copy())
#     draw = ImageDraw.Draw(img, "RGBA")
#     color_rgb = {"red":(220,50,50,220), "blue":(36,100,240,220), "green":(120,200,120,220), "yellow":(245,212,96,220)}
#     id_to_coord = {n["id"]:(n["x"], n["y"]) for n in nodes}
#     # draw all edges
#     for e in edges:
#         u=e["from"]; v=e["to"]
#         if u in id_to_coord and v in id_to_coord:
#             x1,y1 = id_to_coord[u]; x2,y2 = id_to_coord[v]
#             draw.line([(x1,y1),(x2,y2)], fill=color_rgb[e["color"]][:3]+(180,), width=10)
#     # highlight path
#     if path_nodes and len(path_nodes)>1:
#         for a,b in zip(path_nodes, path_nodes[1:]):
#             if a in id_to_coord and b in id_to_coord:
#                 x1,y1 = id_to_coord[a]; x2,y2 = id_to_coord[b]
#                 draw.line([(x1,y1),(x2,y2)], fill=(255,0,255,255), width=14)
#     # draw nodes
#     for n in nodes:
#         x,y = n["x"], n["y"]; r = 10
#         draw.ellipse([x-r,y-r,x+r,y+r], fill=(0,0,0,255))
#     img.save(out_path)
#     return out_path




# visualize.py
import cv2
import numpy as np
from typing import List, Dict

def draw_overlay(rgb: np.ndarray, color_masks: Dict[str, np.ndarray],
                 skel: np.ndarray, nodes: List[Dict], edges: List[Dict],
                 path: List[str], out_image: str):
    """
    Highlight only the selected path while keeping original maze colors intact.
    Dim all other areas.
    """
    H, W, _ = rgb.shape
    overlay = rgb.copy()

    # 1. Create mask for path pixels
    path_mask = np.zeros((H, W), dtype=np.uint8)
    node_map = {n['id']: n for n in nodes}

    # Collect edges along the path
    path_edges = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        for e in edges:
            if (e['from'] == u and e['to'] == v) or (e['from'] == v and e['to'] == u):
                path_edges.append(e)
                break

    for e in path_edges:
        pts = np.array(e['pts'], dtype=np.int32)
        for p in pts:
            x, y = p
            for yy in range(max(0, y-1), min(H, y+2)):
                for xx in range(max(0, x-1), min(W, x+2)):
                    path_mask[yy, xx] = 255

    # 2. Dim all non-path pixels without converting to grayscale
    dim_factor = 0.25  # reduce intensity of non-path pixels
    overlay = (overlay * dim_factor).astype(np.uint8)
    overlay[path_mask == 255] = rgb[path_mask == 255]  # restore original color for path

    # 3. Draw path edges in bright green (optional)
    for e in path_edges:
        pts = np.array(e['pts'], dtype=np.int32)
        for j in range(len(pts)-1):
            cv2.line(overlay, tuple(pts[j]), tuple(pts[j+1]), (0,255,0), 2)  # bright green

    # 4. Draw nodes along the path in red
    for nid in path:
        n = node_map[nid]
        x, y = int(n['x']), int(n['y'])
        cv2.circle(overlay, (x,y), 6, (0,0,255), -1)  # red nodes

    # 5. Save output
    cv2.imwrite(out_image, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))