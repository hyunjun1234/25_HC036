#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gallery DB builder with DeGirum SCRFD + ArcFace on Hailo-8L (@local).
- Input: image files in --gallery (e.g., byeongchan_01.jpg, hyunjun_01.jpg ...)
- Output: LanceDB at --db (table 'face' with 512-D vectors, cosine)
"""
import os, glob, uuid, argparse
import numpy as np
import cv2
import lancedb
from lancedb.pydantic import LanceModel, Vector
import degirum as dg

ARC_REF_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

class FaceRec(LanceModel):
    id: str
    vector: Vector(512)
    entity_name: str

def align112(img_bgr, kps5):
    src = np.array(kps5, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ARC_REF_5PTS, ransacReprojThreshold=1000)
    return cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="face")
    args = ap.parse_args()

    det = dg.load_model(args.det, "@local", args.zoo, "", overlay_color=(0,255,0))
    rec = dg.load_model(args.rec, "@local", args.zoo, "")

    db = lancedb.connect(args.db)
    tbl = db.create_table(args.table, schema=FaceRec) if args.table not in db.table_names() else db.open_table(args.table)

    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    files=[]
    for ex in exts:
        files += glob.glob(os.path.join(args.gallery, ex))
        files += glob.glob(os.path.join(args.gallery, "*", ex))

    if not files:
        print(f"[DB] no images under {args.gallery}")
        return

    added=0
    for path in files:
        base = os.path.basename(path)
        label = base.split("_")[0].split(".")[0]
        out = det(path)
        faces = out.results
        if len(faces) != 1:
            print(f"[DB] skip {base} faces={len(faces)}"); continue
        kps = [lm["landmark"] for lm in faces[0]["landmarks"]]
        aligned = align112(out.image, kps)
        emb = rec(aligned).results[0]["data"][0]
        recrow = FaceRec(id=str(uuid.uuid4()), vector=np.asarray(emb, np.float32), entity_name=label)
        tbl.add([recrow]); added += 1
        print(f"[DB] add {label}: {base}")

    print(f"[DB] added={added}, total={tbl.count_rows()}")

if __name__ == "__main__":
    main()
