# AIROS

# 📝[AIROS - [25_HC036]AI 기반 자율 순찰 및 침입자 감지 경비 로봇]

## **💡1. 프로젝트 개요**

**1-1. 프로젝트 소개**
- 프로젝트 명 : AI 기반 자율 순찰 및 침입자 감지 경비 로봇
- 프로젝트 정의 :  AI 기반 자율 순찰 및 안면 인식 구분을 활용한 침입자 감지 시스템
  <img width="618" height="379" alt="Image" src="https://github.com/user-attachments/assets/e646336e-9d91-4759-9311-3307caa1bacf" /></br>

**1-2. 개발 배경 및 필요성**
- 기존의 CCTV 및 경비 인력 중심 보안 시스템은 사각지대 발생, 높은 인건비 부담, 긴급 상황 시 실시간 대응의 어려움 등 여러 한계를 지니고 있습니다. 본 프로젝트는 이러한 문제를 해결하기 위해 ROS2 및 SLAM 기반 자율주행 기술과 AI 얼굴 인식(ArcFace+MFN) 기술을 융합한 무인 경비 로봇을 개발하고자 합니다. SCRFD 기반 허가자 DB를 통해 간편한 운용이 가능하며, 24시간 무인 감시로 인건비를 절감할 수 있습니다. 또한 장애물 회피 및 경로 재탐색 기능을 통해 자율적 대응이 가능하고, 사족보행 로봇 CANINE을 활용하여 거친 지형에서도 운용할 수 있습니다. 따라서 이러한 차세대 무인 보안 시스템이 필요합니다.


**1-3. 프로젝트 특장점**
- 기존 보안 제품(CCTV, 유인 경비 등)과 비교한 차별성
- 유사 제품 (기존의 단순 무인 경비 로봇)과 비교한 기술적 차별성
- 기존 프로젝트에서 업그레이드된 차별성 (기존 프로젝트 대비)


**1-4. 주요 기능**
- 얼굴감지기능 : SCRFD 모델을 활용해 실시간 영상 스트림에서 사람 얼굴 인식 및 112 X   112 pixel 크기로 크롭, 인식된 사람 얼굴에 bounding box 형성
- 얼굴 크롭 및 DB 구조 자동화 : - Kmeans clustering 알고리즘 활용하여 분산된 임베딩 값 중 대표 이미지 지정 후 파일 이름 자동 변경, MBF 모델을 활용하여 각 사진별 임베딩 추출하여 데이터 파일로 변환 후 face_db에 저장
- 안면 인식 구분 기능 : 라즈베리파이 카메라로 얼굴 촬영, 촬영된 이미지를 MBF 모델을 hef 변환 파일을 활용하여 hailort로 실시간 임베딩 추출, 추출된 임베딩을 가지고 기존 DB에 사람별로 저장된 임베딩과 비교하여 얼굴 인식
- 맵 그리기 및 경로 순찰 : LiDAR 센서 데이터를 기반으로 실내 SLAM 맵을 생성 및 지정된 포인트로 이동하는 경로 계획(WayPoint Following) 기능 수행. 장애물을 실시간으로 회피하며 최적의 경로로 자율 주행
- MQTT 알림 기능 : 비허가자 감지시 MQTT 프로토콜을 통해 관리자에게 실시간 알림 발송

**1-5. 기대 효과 및 활용 분야**
- 기대 효과 
1. 온디바이스 AI 기반 자율 경비 로봇으로 스마트 보안 시장 개척 제공.
2. Hailo-8L 활용한 실시간 객체·얼굴 인식으로 즉각 대응력 제공.
3. ROS2·SLAM 기술을 통한 자율 경로 학습 및 안정적 이동 제공.
4. 저비용 엣지 디바이스 활용으로 경제적 구축 및 인건비 절감 제공.
5. 출입 관리·근태 체크 등 확장성 높은 보안 솔루션 제공.

- 활용 분야 :
1. 군부대·주요 시설의 무인 경계 순찰로 보안 효율 향상 및 인력 비용 절감 제공.
2. 병원·연구소 등에서 실시간 침입 탐지와 즉각 대응을 통한 보안 강화 제공.
3. 얼굴 인식 기반 출입 관리·근태 체크 등 산업 현장 운영 효율성 향상 제공.
4. 스마트홈·헬스케어·리테일 등 생활 분야까지 확장 가능한 AI 활용성 제공.

**1-6. 기술 스택**
- 프론트엔드 : React, Next.js, Tailwind CSS
- 백엔드 : Python(FastAPI), Node.js, Django
- AI/ML : PyTorch, TensorFlow, Hugging Face, OpenAI API
- 데이터베이스 : PostgreSQL, MongoDB, Elasticsearch
- 클라우드 : notion
- 배포 및 관리 : GitHub Actions

- AI / 머신러닝 
프레임워크 : HailoRT SDK, DeGirum pySDK

- 사용 모델 : 
Face Detection: SCRFD (10G, 2.5G, Hailo HEF)
Face Recognition: ArcFace (MobileFaceNet 백본, HEF / ONNX)
Clustering: k-means / medoid 기반 대표 임베딩 선정
후처리: Cosine Similarity 기반 갤러리 매칭
임베딩 DB: LanceDB (512-D vector 검색)

- 임베디드 / 온디바이스 AI
보드: Raspberry Pi 5 + Hailo 8L NPU (PCIe)
운영체제: Ubuntu 24.04 (Noble)
드라이버 / SDK:
HailoRT 4.17 – 4.20
TAPPAS 3.31.0
Insightface

- 로보틱스 / ROS2 시스템
플랫폼: ROS 2 Jazzy + Nav2 + Cartographer SLAM
센서: 2D LiDAR (SLLIDAR C1), Camera (libcamera)
자율주행: Waypoint navigation loop, Obstacle avoidance

---

## **💡2. 팀원 소개**
| <img width="80" height="100" src="https://github.com/user-attachments/assets/ab73bb1c-c1d4-464d-8ad3-635b45d5a8ae" > | <img width="80" height="100" alt="image" src="https://github.com/user-attachments/assets/c7f66b7c-ab84-41fa-8fba-b49dba28b677" > | <img width="80" height="100" alt="image" src="https://github.com/user-attachments/assets/c33252c7-3bf6-43cf-beaa-a9e2d9bd090b" > | <img width="80" height="100" alt="image" src="https://github.com/user-attachments/assets/0d5909f0-fc73-4ab9-be09-4d48e3e71083" > | <img width="80" height="100" alt="image" src="https://github.com/user-attachments/assets/c7f66b7c-ab84-41fa-8fba-b49dba28b677" > |
|:---:|:---:|:---:|:---:|:---:|
| **조현준** | **양병찬** | **김태훈** | **강신재** | **권용인** |
| • 개발총괄 <br> • 안면 구분 시스템 개발 | • 안면 구분 시스템 개발 • 후처리 | • 자율주행 시스템 개발 |• MQTT 개발 <br> • 서버 구축 | • 프로젝트 멘토 <br> • 기술 자문 |



---
## **💡3. 시스템 구성도**
> **(참고)** S/W구성도, H/W구성도, 서비스 흐름도 등을 작성합니다. 시스템의 동작 과정 등을 추가할 수도 있습니다.
- 서비스 구성도
<img width="618" height="379" alt="Image" src="https://github.com/user-attachments/assets/e646336e-9d91-4759-9311-3307caa1bacf" /></br>


- 알고리즘 흐름도
<img width="555" height="358" alt="Image" src="https://github.com/user-attachments/assets/be35b0ce-0d9f-4e6f-acd9-4d12e415dc8f" /></br>

<img width="581" height="329" alt="Image" src="https://github.com/user-attachments/assets/0d80736a-c8ab-4cfc-ac98-f00e3f794427" />

---
## **💡4. 작품 소개영상**
> **참고**: 썸네일과 유튜브 영상을 등록하는 방법입니다.
```Python
아래와 같이 작성하면, 썸네일과 링크등록을 할 수 있습니다.
[![영상 제목](유튜브 썸네일 URL)](유튜브 영상 URL)

작성 예시 : 저는 다음과 같이 작성하니, 아래와 같이 링크가 연결된 썸네일 이미지가 등록되었네요! 
[![한이음 드림업 프로젝트 소개](https://github.com/user-attachments/assets/16435f88-e7d3-4e45-a128-3d32648d2d84)](https://youtu.be/YcD3Lbn2FRI?si=isERqIAT9Aqvdqwp)
```
[![아이로스 자율주행 로봇](https://img.youtube.com/vi/QY_Du2YTDjE/0.jpg)](https://youtu.be/QY_Du2YTDjE?si=FAy1_ID-kJWe4QAW)



---
## **💡5. 핵심 소스코드**

- 소스코드 설명 : 비교 임베딩 추출 코드입니다.
1. 얼굴 정렬 이미지(112×112) → ArcFace-MobileFaceNet 임베딩 추출
2. 임베딩 → K-means 기반 대표 임베딩 추출 및 갤러리에 저장
```python
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

```



- 소스코드 설명 : rpi 카메라로 촬영한 실시간 얼굴과 db의 임베딩을 비교하여 안면 인식을 하는 코드입니다.
1. 실시간 영상에서 얼굴 크롭 후 임베딩 추출
2. 추출된 임베딩과 갤러리에 저장되어 있던 임베딩을 비교하기 위해 코사인 유사도 계산 → 3.임계값/마진/다수결/스무딩 적용
3. 최종 ID 확정 또는 Unknown 처리 → 결과 OSD로 프레임에 합성
4. 출력: 터미널에 주기적 로그 송출

```python
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIFO(YUV420) -> SCRFD(det + 5pts, DeGirum@local) -> ArcFace(112, DeGirum@local)
-> LanceDB cosine match -> OSD (+ Unknown dwell detection)
- 터미널 A: ./cam_start.sh (FIFO 생성 및 카메라 캡쳐)  [예: /tmp/cam.yuv, 640x480@30]
- 터미널 B: python3 fifo_scrfd_face_match_osd.py --zoo ... --fifo /tmp/cam.yuv ...
"""
import os, sys, time, argparse, signal
import numpy as np
import cv2
import lancedb
from collections import deque, Counter
import degirum as dg

# ---------- 기본값 ----------
ARC_REF_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

def read_exact(f, n):
    buf = bytearray(n); mv = memoryview(buf); off = 0
    while off < n:
        r = f.readinto(mv[off:])
        if not r: return None
        off += r
    return buf

def align112(img_bgr, kps5):
    src = np.array(kps5, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ARC_REF_5PTS, ransacReprojThreshold=1000)
    return cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0)

def l2_normalize(x, eps=1e-9):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)

# ----- LanceDB -----
def open_table(db_uri, table="face"):
    db = lancedb.connect(db_uri)
    return db.open_table(table)

class TrackSmoother:
    def __init__(self, thresh=0.45, smooth=7):
        self.hist = deque(maxlen=max(1,int(smooth)))
        self.thresh = float(thresh)
    def update(self, name, sim):
        self.hist.append((name, float(sim)))
        cnt = Counter([n for n,_ in self.hist]).most_common(1)[0]
        mode = cnt[0]
        avg = float(np.mean([s for n,s in self.hist if n==mode]))
        return (mode if avg>=self.thresh else "unknown"), avg

class SimpleTracker:
    def __init__(self, dist_th=80, max_miss=15):
        self.next_id = 0
        self.tr = {}  # id -> dict(x,y,w,h,miss,sm,label,avg,last_seen)
        self.dist_th = dist_th; self.max_miss = max_miss
    @staticmethod
    def _c(b): return (b[0]+b[2]/2.0, b[1]+b[3]/2.0)
    def update(self, boxes):
        for t in self.tr.values(): t["miss"] += 1
        used = set()
        for b in boxes:
            cx, cy = self._c(b)
            j=None; best=1e9
            for tid,t in self.tr.items():
                tx, ty = self._c((t["x"],t["y"],t["w"],t["h"]))
                d = (tx-cx)**2 + (ty-cy)**2
                if d<best and tid not in used:
                    j, best = tid, d
            if j is not None and best**0.5 < self.dist_th:
                t = self.tr[j]; t.update({"x":int(b[0]),"y":int(b[1]),"w":int(b[2]),"h":int(b[3]),"miss":0})
                used.add(j)
            else:
                tid=self.next_id; self.next_id+=1
                self.tr[tid]={"x":int(b[0]),"y":int(b[1]),"w":int(b[2]),"h":int(b[3]),
                              "miss":0,"sm":None,"label":"unknown","avg":0.0,"last_seen":time.time()}
        # drop
        for tid in [k for k,v in self.tr.items() if v["miss"]>self.max_miss]:
            del self.tr[tid]
        return self.tr

# ─────────────────────────────
# 워밍업 (기존 성능 최적화 유지)
# ─────────────────────────────
def warmup(det, rec, w, h, n=5):
    dummy = np.zeros((h, w, 3), np.uint8)
    for _ in range(n):
        out = det(dummy)
        if out.results:
            kps = [lm["landmark"] for lm in out.results[0]["landmarks"]]
            aligned = align112(dummy, kps)
        else:
            aligned = cv2.resize(dummy, (112,112))
        _ = rec(aligned)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True, help="DeGirum local zoo folder (parent of model packages)")
    ap.add_argument("--det", required=True, help="SCRFD model folder name")
    ap.add_argument("--rec", required=True, help="ArcFace-MobileFaceNet model folder name")
    ap.add_argument("--fifo", default="/tmp/cam.yuv")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--db", required=True)
    ap.add_argument("--table", default="face")
    ap.add_argument("--thresh", type=float, default=0.40)
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--det-every", type=int, default=1)
    ap.add_argument("--min-face", type=int, default=80)
    ap.add_argument("--flip", type=int, default=None, help="-1(180),0(상하),1(좌우)")
    # [ADDED] 외부인 체류 감지 옵션
    ap.add_argument("--margin", type=int, default=40, help="pixels from border regarded as edge zone")               # [ADDED]
    ap.add_argument("--unknown-time", type=float, default=3.0, help="seconds for unknown dwell to trigger alert")    # [ADDED]
    ap.add_argument("--reset-time", type=float, default=2.0, help="seconds after no unknown-in-margin to reset")     # [ADDED]
    args = ap.parse_args()

    # DeGirum models (@local)
    det = dg.load_model(args.det, "@local", args.zoo, "", overlay_color=(0,255,0))
    rec = dg.load_model(args.rec, "@local", args.zoo, "")

    # DB
    tbl = open_table(args.db, args.table)

    # 워밍업
    warmup(det, rec, args.w, args.h, n=5)

    # FIFO
    frame_size = args.w * args.h * 3 // 2  # YUV420
    print(f"[FIFO] open {args.fifo} ({args.w}x{args.h})")
    f = open(args.fifo, "rb", buffering=0)

    # 초기 프레임 드롭(버퍼 안정)
    DROP_N = 10
    for _ in range(DROP_N):
        raw = read_exact(f, frame_size)
        if raw is None: break

    tracker = SimpleTracker(dist_th=90, max_miss=12)

    # 슬라이딩 윈도우 FPS
    times = deque(maxlen=120)
    disp_fps = 0.0
    frames = 0

    # [ADDED] 외부인(unknown) 체류 감지 상태
    unknown_timer = {}   # tid -> {"start":float or None, "last":float}                              # [ADDED]
    alert_active = False                                                                     # [ADDED]
    last_unknown_in_margin_time = 0.0                                                        # [ADDED]

    signal.signal(signal.SIGINT, signal.SIG_DFL)

    while True:
        raw = read_exact(f, frame_size)
        if raw is None: break
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(args.h*3//2, args.w)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        if args.flip is not None:
            bgr = cv2.flip(bgr, args.flip)

        # ── SCRFD 검출 ────────────────────────────────────────────────────────
        boxes=[]; kps_list=[]
        if frames % max(1,args.det_every) == 0:
            out = det(bgr)  # numpy BGR 입력 OK
            for face in out.results:
                (x1,y1,x2,y2) = map(int, face["bbox"])
                w = max(0, x2-x1); h = max(0, y2-y1)
                if w < args.min_face or h < args.min_face: continue
                boxes.append((x1,y1,w,h))
                kps_list.append([lm["landmark"] for lm in face["landmarks"]])
        # ─────────────────────────────────────────────────────────────────────

        # ── 배치 정합 + 배치 임베딩 + 매칭 ─────────────────────────────────────
        box_labels, box_sims = [], []
        if boxes:
            aligned_list = [align112(bgr, kps) for kps in kps_list]
            embeddings = []
            for emb_res in rec.predict_batch(aligned_list):
                embeddings.append(emb_res.results[0]["data"][0])
            for q in embeddings:
                res = (tbl.search(np.asarray(q, np.float32), vector_column_name="vector")
                         .metric("cosine").limit(1).to_list())
                if res:
                    sim = float(1.0 - res[0]["_distance"])
                    name = res[0]["entity_name"]
                else:
                    sim, name = 0.0, "unknown"
                box_labels.append(name); box_sims.append(sim)
        # ─────────────────────────────────────────────────────────────────────

        # ── 트래커 업데이트 ───────────────────────────────────────────────────
        tracks = tracker.update(boxes)

        # 박스 ↔ 트랙 매칭(가장 가까운 중심 매칭)
        def nearest_box_index_to_track(tr, boxes_list):
            if not boxes_list: return None
            cx = tr["x"] + tr["w"]/2.0; cy = tr["y"] + tr["h"]/2.0
            j=None; best=1e18
            for i,(x,y,w,h) in enumerate(boxes_list):
                bx = x + w/2.0; by = y + h/2.0
                d = (bx-cx)**2 + (by-cy)**2
                if d < best: best = d; j = i
            return j

        now = time.time()  # [ADDED] 체류 시간 판정용
        any_unknown_in_margin = False  # [ADDED]

        for tid,t in tracks.items():
            if t["sm"] is None: t["sm"] = TrackSmoother(args.thresh, args.smooth)

            idx = nearest_box_index_to_track(t, boxes)
            if idx is not None and idx < len(box_labels):
                name = box_labels[idx]; sim = box_sims[idx]
            else:
                name = "unknown"; sim = 0.0

            label, avg = t["sm"].update(name, sim)
            t["label"], t["avg"], t["last_seen"] = label, avg, time.time()

            # [ADDED] ─────────────────────────────────────────────────────────
            # 외부인(unknown) 체류 감지: 프레임 경계 margin 내에 unknown이 일정 시간 이상 머무르면 alert
            x,y,w,h = t["x"], t["y"], t["w"], t["h"]
            in_margin = (x <= args.margin or y <= args.margin or
                         (x+w) >= (args.w - args.margin) or
                         (y+h) >= (args.h - args.margin))
            is_unknown = (t["label"] == "unknown")

            if is_unknown and in_margin:
                any_unknown_in_margin = True
                timer = unknown_timer.get(tid, {"start": None, "last": now})
                if timer["start"] is None:
                    timer["start"] = now
                timer["last"] = now
                unknown_timer[tid] = timer
                dwell = now - timer["start"]
                if dwell >= args.unknown_time:
                    alert_active = True
            else:
                # margin 밖이거나 라벨이 바뀐 경우: 타이머 유지하되 last만 갱신
                if tid in unknown_timer:
                    unknown_timer[tid]["last"] = now
            # ─────────────────────────────────────────────────────────────────

        # [ADDED] alert reset 로직
        if any_unknown_in_margin:
            last_unknown_in_margin_time = now
        else:
            if alert_active and (now - last_unknown_in_margin_time) >= args.reset_time:
                alert_active = False
                unknown_timer.clear()

        # ── OSD & FPS ────────────────────────────────────────────────────────
        frames += 1
        pnow = time.perf_counter()
        times.append(pnow)
        while times and pnow - times[0] > 1.0:
            times.popleft()
        if len(times) >= 2:
            disp_fps = (len(times)-1) / (times[-1] - times[0])

        canvas = bgr

        # [ADDED] margin 가이드 라인
        if args.margin > 0:
            cv2.rectangle(canvas, (args.margin, args.margin), (args.w-args.margin, args.h-args.margin),
                          (50,50,50), 1, cv2.LINE_AA)

        for tid,t in tracks.items():
            x,y,w,h = t["x"],t["y"],t["w"],t["h"]
            color = (0,255,0) if t["label"]!="unknown" else (0,0,255)
            cv2.rectangle(canvas, (x,y), (x+w,y+h), color, 2)
            cv2.putText(canvas, f"ID{tid}:{t['label']} {t['avg']:.2f}",
                        (x, max(18,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            # [ADDED] unknown dwell 시간 OSD
            if tid in unknown_timer and t["label"]=="unknown":
                dwell = now - (unknown_timer[tid]["start"] or now)
                cv2.putText(canvas, f"dwell {dwell:.1f}s",
                            (x, y+h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2, cv2.LINE_AA)

        # 헤더 바
        cv2.rectangle(canvas, (10,10), (args.w-10, 55), (0,0,0), -1)
        head = f"{args.w}x{args.h} FPS~{disp_fps:.1f}"
        # [ADDED] alert state 출력
        head2 = f"  |  value={1 if alert_active else 0}  (unk>= {args.unknown_time:.1f}s in margin, reset {args.reset_time:.1f}s)"
        cv2.putText(canvas, head + head2, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,255) if not alert_active else (0,128,255), 2, cv2.LINE_AA)

        cv2.imshow("LIVE (FIFO + SCRFD + ArcFace)", canvas)
        if cv2.waitKey(1) & 0xFF == 27: break

    f.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

```
