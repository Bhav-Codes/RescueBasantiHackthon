# main.py
import argparse
import numpy as np
from src.image_to_graph import load_image_rgb, create_color_masks, skeletonize_fallback, detect_black_circles, extract_graph_from_skeleton
from src.coolant_path import dijkstra_deterministic
from src.visualize import draw_overlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to plant image (PNG/JPG).")
    ap.add_argument("--scale", type=float, default=50.0, help="Pixels-per-unit (default 50).")
    ap.add_argument("--start", help="Start node id (optional). If omitted, auto-mapped from black marker).")
    ap.add_argument("--target", help="Target node id (optional).")
    ap.add_argument("--out-json", default="extracted_graph.json")
    ap.add_argument("--out-image", default="overlay_highlighted.png")
    args = ap.parse_args()

    rgb = load_image_rgb(args.image)
    masks = create_color_masks(rgb)
    combined = None
    for m in masks.values():
        combined = m if combined is None else (combined | m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    filled = cv2.morphologyEx((combined>0).astype(np.uint8)*255, cv2.MORPH_CLOSE, kernel)
    skel = skeletonize_fallback((filled>0).astype(np.uint8))
    circles = detect_black_circles(rgb)
    nodes, edges = extract_graph_from_skeleton(skel, masks, scale_pixels_per_unit=args.scale)

    # map circles to nearest nodes for start & target if not provided
    start = args.start; target = args.target
    if (not start or not target) and len(circles) >= 2 and nodes:
        # choose leftmost circle as start, rightmost as target
        start_circle = circles[0]; target_circle = circles[-1]
        def nearest(px,py):
            best=None; bestd=1e9
            for n in nodes:
                d = ((n["x"]-px)**2 + (n["y"]-py)**2)**0.5
                if d < bestd:
                    bestd = d; best = n["id"]
            return best
        if not start: start = nearest(start_circle["x"], start_circle["y"])
        if not target: target = nearest(target_circle["x"], target_circle["y"])

    path, total = dijkstra_deterministic(nodes, edges, start, target)
    print("Fastest Path:", path)
    print("Total Time (seconds):", total)
    import json
    with open(args.out_json, "w") as f:
        json.dump({"scale_pixels_per_unit": args.scale, "nodes": nodes, "edges": edges}, f, indent=2)
    draw_overlay(rgb, masks, skel, nodes, edges, path, args.out_image)

if __name__ == "__main__":
    import cv2
    main()