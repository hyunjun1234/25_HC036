#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIFO(YUV420) -> SCRFD(det + 5pts, DeGirum@local) -> ArcFace(112, DeGirum@local)
-> LanceDB cosine match -> OSD (+ Unknown dwell detection)
- 터미널 A: ./cam_start.sh (FIFO 생성 및 카메라 캡쳐)  [예: /tmp/cam.yuv, 640x480@30]
- 터미널 B: python3 fifo_scrfd_face_match_osd.py --zoo ... --fifo /tmp/cam.yuv ...
"""
import os, sys, time, argparse, signal
sys.path.append("/home/pinky/hailo_examples/openpath-CANINE-middleware/python_scripts")
import numpy as np
import rclpy
from rosCommunication import ROSCommunication
import cv2
import lancedb
from collections import deque, Counter
import degirum as dg
from canineStruct import Command
from sharedMemory import SharedMemoryManager
import threading
import subprocess

def rosCommunicationThread(shm, args=None):
    rclpy.init(args=args)
    node = ROSCommunication(shm)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

_seq_running = False
_seq_lock = threading.Lock()

def _send_dwa_noinput_sequence(shm, tag="SEQ"):
    """비차단: DWA -> 3s -> NO_INPUT -> 3s 시퀀스 1회 발사"""
    global _seq_running
    with _seq_lock:
        if _seq_running:
            print(f"[FACE] {tag}: sequence already running, skip")
            return
        _seq_running = True

    def _worker():
        global _seq_running
        try:
            print(f"[FACE] {tag}: send DWA")
            shm.cmd.command = Command.DWA.value
            time.sleep(2.0)

            print(f"[FACE] {tag}: send NO_INPUT")
            shm.cmd.command = Command.NO_INPUT.value
            time.sleep(2.0)

            print(f"[FACE] {tag}: sequence done")
        finally:
            with _seq_lock:
                _seq_running = False

    threading.Thread(target=_worker, daemon=True).start()
    
#경고음 재생
_alarm_proc = None
def start_alarm(path):
    """경고음 재생 시작 (이미 재생 중이면 무시)"""
    global _alarm_proc
    if _alarm_proc and _alarm_proc.poll() is None:
        return
    # aplay 고정 (WAV 권장). MP3 쓰면 mpg123로 교체 가능.
    _alarm_proc = subprocess.Popen(["mpg123", "-q", path])

def stop_alarm():
    """경고음 중지 (재생 중일 때만)"""
    global _alarm_proc
    if _alarm_proc and _alarm_proc.poll() is None:
        try:
            _alarm_proc.terminate()
        except Exception:
            pass
    _alarm_proc = None

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
    ap.add_argument("--h", type=int, default=640)
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
    ap.add_argument("--ros-wait", type=float, default=8.0, help="seconds to wait middleware_connected")              # [ADD]
    ap.add_argument("--no-clear", action="store_true", help="do not send NO_INPUT on alert clear")
    # 벨 울리기
    ap.add_argument("--alarm-file", required=True, help="경고음 파일 경로 (예: /home/pinky/hailo_examples/assets/alarm.wav)")
    args = ap.parse_args()
    
    shm = SharedMemoryManager()
    
    threading.Thread(target=rosCommunicationThread, args=(shm,), daemon=True).start()
    
    t0 = time.time()
    while not shm.middleware_connected and (time.time() - t0) < args.ros_wait:
        time.sleep(0.2)
    print(f"[FACE] middleware_connected={shm.middleware_connected}  (waited {time.time()-t0:.1f}s)")

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
    prev_alert = False
    last_send_ts = 0.0   
    MIN_SEND_INTERVAL = 1.0   
    last_sent_cmd = None

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    prev_mw = None           # 이전 middleware_connected
    prev_alert_state = None  # 이전 alert_active
    
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
            # 외부인(unknown) 체류 감지: 프레임 경계 사각형 내에 unknown이 일정 시간 이상 머무르면 alert
            x,y,w,h = t["x"], t["y"], t["w"], t["h"]
            in_margin = (x >= args.margin and y >= args.margin and
                         (x+w) <= (args.w - args.margin) and
                         (y+h) <= (args.h - args.margin))
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
        # ─────────────────────────────────────────────────────────
        
        mw_now = bool(shm.middleware_connected)
        alert_now = bool(alert_active)

        # 알람 '변화' 먼저 체크
        if mw_now and (prev_alert_state is not None) and (alert_now != prev_alert_state):
            if alert_now:
                #경보음 울리기
                start_alarm(args.alarm_file)
                _send_dwa_noinput_sequence(shm, tag="ALERT_ON")
            else:
                stop_alarm()
                _send_dwa_noinput_sequence(shm, tag="ALERT_OFF")

        # 그 다음에 상태 로그 + 상태 저장
        if (mw_now != prev_mw) or (alert_now != prev_alert_state):
            print(f"[FACE] middleware_connected={mw_now}  alert={alert_now}")
        prev_mw = mw_now
        prev_alert_state = alert_now

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
        # [ADD] 현재 전송 상태(연결/보낸 명령 최근 시각) 간단 표시
        cmd_hint = ""
        try:
            cur_cmd = getattr(shm.cmd, "command", None)
            if cur_cmd is not None:
                cmd_hint = f"  |  CMD={Command(cur_cmd).name}"
        except Exception:
            pass
        cv2.putText(canvas, head + head2 + cmd_hint, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,255) if not alert_active else (0,128,255), 2, cv2.LINE_AA)

        cv2.imshow("LIVE (FIFO + SCRFD + ArcFace)", canvas)
        if cv2.waitKey(1) & 0xFF == 27: break

    f.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
