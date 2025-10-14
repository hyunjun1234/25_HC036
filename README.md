# AIROS

# 📝[AIROS - [25_HC036]AI 기반 자율 순찰 및 침입자 감지 경비 로봇]

## **💡1. 프로젝트 개요**

**1-1. 프로젝트 소개**
- 프로젝트 명 : AI 기반 자율 순찰 및 침입자 감지 경비 로봇
- 프로젝트 정의 :  AI 기반 자율 순찰 및 안면 인식 구분을 활용한 침입자 감지 시스템
  <img width="618" height="379" alt="Image" src="https://github.com/user-attachments/assets/71bac2d7-f7e6-4ed8-bea6-3ae301278a0b" /></br>

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
| <img width="80" height="100" src="https://github.com/user-attachments/assets/975a5d97-1810-4247-a80c-12293b44ec02" > | <img width="80" height="100" alt="image" src="https://github.com/user-attachments/assets/74897ec8-90c3-459c-be7b-ffe76398d7ba" > | <img width="80" height="100" alt="image" src="https://github.com/user-attachments/assets/65efd625-3665-4f1f-8bf1-d7f34694242b" > | <img width="80" height="100" alt="image" src="https://github.com/user-attachments/assets/3362de4a-d1e2-474f-9e26-31a690fd2b65" > | <img width="80" height="100" alt="image" src="https://github.com/user-attachments/assets/30af478e-0c01-463f-afcd-dbc7da8e9cb9" > |
|:---:|:---:|:---:|:---:|:---:|
| **조현준** | **양병찬** | **김태훈** | **강신재** | **권용인** |
| • 개발총괄 <br> • 안면 구분 시스템 개발 | • 안면 구분 시스템 개발 <br>• 후처리 | • 자율주행 시스템 개발 |• MQTT 개발 <br> • 서버 구축 | • 프로젝트 멘토 <br> • 기술 자문 |



---
## **💡3. 시스템 구성도**
> 서비스 구성도

<img width="618" height="379" alt="Image" src="https://github.com/user-attachments/assets/e646336e-9d91-4759-9311-3307caa1bacf" /></br>


> 알고리즘 흐름도

<img width="555" height="358" alt="Image" src="https://github.com/user-attachments/assets/be35b0ce-0d9f-4e6f-acd9-4d12e415dc8f" /></br>

<img width="581" height="329" alt="Image" src="https://github.com/user-attachments/assets/0d80736a-c8ab-4cfc-ac98-f00e3f794427" />

---
## **💡4. 작품 소개영상**

[![AI 기반 자율 순찰 및 침입자 감지 경비 로봇](https://github.com/user-attachments/assets/4cb84602-90a7-48f5-a27f-40256d9be833)](https://youtu.be/HwIvVmLQAVw)



---
## **💡5. 핵심 소스코드**

### 안면인식 핵심 코드

- 소스코드 설명 : 비교 임베딩 추출 코드입니다.
1. 얼굴 정렬 이미지(112×112) → ArcFace-MobileFaceNet 임베딩 추출
2. 임베딩 → K-means 기반 대표 임베딩 추출 및 갤러리에 저장

### Stage 1: Face Detection with Keypoint Detection

목표: 얼굴 bbox + 5점 랜드마크를 얻어 이후 정렬(Alignment)에 사용합니다.
아래는 SCRFD를 DeGirum @local로 로드하고, 한 프레임에서 검출·랜드마크를 꺼내는 최소 예시입니다.

```python
import degirum as dg

# SCRFD 로드 (@local, 모델은 로컬 zoo 디렉토리 하위)
det = dg.load_model(
    model_name="scrfd_10g--640x640_quant_hailort_hailo8l_1",  # 예시
    inference_host_address="@local",
    zoo_url="/path/to/degirum/models_hailort",
    token=""
)

# BGR 프레임 bgr 에 대해 추론
out = det(bgr)  # numpy BGR 입력 가능

# bbox & keypoints 수집
boxes = []
kps_list = []
for face in out.results:
    x1, y1, x2, y2 = map(int, face["bbox"])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    if w < 80 or h < 80:  # 최소 크기 필터(예시)
        continue
    boxes.append((x1, y1, w, h))
    kps_list.append([lm["landmark"] for lm in face["landmarks"]])  # 5점 랜드마크
```


### Stage 2: Alignment (112×112 정렬)

목표: 랜드마크를 ArcFace 기준 좌표에 맞춰 유사변환으로 정렬합니다.
정렬 후 112×112 크기의 정규화된 얼굴 패치를 임베딩 모델에 입력합니다.

```python
import numpy as np
import cv2

# ArcFace 기준 5점(112×112 기준)
ARC_REF_5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align112(img_bgr, kps5):
    src = np.array(kps5, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(
        src,
        ARC_REF_5PTS,
        ransacReprojThreshold=1000
    )
    aligned = cv2.warpAffine(
        img_bgr,
        M,
        (112, 112),
        borderValue=0
    )
    return aligned

```

### Stage 3: Extracting Embeddings (ArcFace MobileFaceNet)

목표: 정렬된 얼굴로부터 512-D 임베딩을 추출합니다. 배치 추론을 통해 성능을 높입니다.
```python
import degirum as dg
import numpy as np

# ArcFace-MobileFaceNet 로드 (@local)
rec = dg.load_model(
    model_name="arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1",
    inference_host_address="@local",
    zoo_url="/path/to/degirum/models_hailort",
    token=""
)

# 배치 정렬
aligned_list = [align112(bgr, kps) for kps in kps_list]

# 배치 임베딩 추출
embeddings = []
for emb_res in rec.predict_batch(aligned_list):
    emb = emb_res.results[0]["data"][0]  # 512-D
    embeddings.append(emb)

# (선택) L2 정규화 → 코사인 유사도에 적합
def l2_normalize(x, eps=1e-9):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)

embeddings = [l2_normalize(np.asarray(e, np.float32)) for e in embeddings]
```

### Stage 4: Database Matching (LanceDB, Cosine)
목표: 임베딩을 LanceDB에 질의하여 가장 가까운 인물(라벨)을 반환합니다.
테이블은 vector(512)와 entity_name 컬럼을 가지는 것으로 가정합니다.

코사인 유사도에서 1 - distance로 유사도를 계산합니다.
thresh를 적절히 잡아 Unknown 판정을 제어할 수 있습니다.
```python
import lancedb
import numpy as np

# DB 접속 및 테이블 열기
db = lancedb.connect(uri="/path/to/face_database")
tbl = db.open_table("face")  # 스키마: id, vector(512), entity_name

def identify_one(embedding, tbl, field="vector", metric="cosine", thresh=0.40):
    res = (tbl.search(np.asarray(embedding, np.float32), vector_column_name=field)
             .metric(metric)
             .limit(1)
             .to_list())
    if not res:
        return "Unknown", 0.0
    sim = float(1.0 - res[0]["_distance"])  # cosine similarity
    name = res[0]["entity_name"] if sim >= thresh else "Unknown"
    return name, sim

# 예: 배치 결과 매칭
labels, sims = [], []
for q in embeddings:
    name, sim = identify_one(q, tbl, field="vector", metric="cosine", thresh=0.40)
    labels.append(name)
    sims.append(sim)
```

### Running on a Video Stream (FIFO + Tracker + Unknown Dwell Alert)

목표: 실시간 YUV420(FIFO) 스트림에서 검출·정렬·임베딩·매칭을 수행하고,
간단한 트래킹/스무딩과 Unknown 체류 경보(value=1) 를 구현합니다.

화면에 미리 정한 감시 구역(ALERT_ROI, 사각형) 안에 unknown 트랙이 3초 이상 머물면 value = 1이 됩니다.
트래커가 사람마다 고유 track id를 유지하므로, 허가자(라벨 있음)와 비허가자(unknown)를 동시에 추적해도 충돌 없이 동작합니다.
허가자 임베딩을 DB에 저장해 두면, 매 프레임 얼굴 인식 → 라벨 부여 → 트래킹 연계가 이루어져 unknown만 정확히 감지합니다.

```python
# 사전 정의: 경보 영역(예: 화면 중앙 320x240)
ALERT_ROI = (W//2 - 160, H//2 - 120, 320, 240)  # (x, y, w, h)
UNKNOWN_DWELL_SEC = 3.0

# 상태값
alert_active = False         # value == 1 이면 True
unknown_timer = {}           # {tid: {"start": float or None, "last": float}}
last_unknown_inside_ts = 0.0
RESET_SEC = 2.0

def inside_roi(box, roi):
    x, y, w, h = box
    rx, ry, rw, rh = roi
    # 중심점이 ROI 안이면 inside로 간주
    cx, cy = x + w/2.0, y + h/2.0
    return (rx <= cx <= rx+rw) and (ry <= cy <= ry+rh)

# 매 프레임마다 실행 (tracks는 SimpleTracker 결과, 각 t에 bbox/label/avg 포함)
now = time.time()
any_unknown_inside = False

for tid, t in tracks.items():
    # t["label"]은 DB 매칭+스무딩 결과 ("unknown" 또는 사람 이름)
    is_unknown = (t["label"] == "unknown")
    in_roi = inside_roi((t["x"], t["y"], t["w"], t["h"]), ALERT_ROI)

    if is_unknown and in_roi:
        any_unknown_inside = True
        timer = unknown_timer.get(tid, {"start": None, "last": now})
        if timer["start"] is None:
            timer["start"] = now
        timer["last"] = now
        unknown_timer[tid] = timer

        dwell = now - timer["start"]
        if dwell >= UNKNOWN_DWELL_SEC:
            alert_active = True  # ← 여기서 value = 1이 됩니다.
    else:
        # ROI 밖이거나 라벨이 바뀐 경우에도 last만 갱신해 흔적 유지
        if tid in unknown_timer:
            unknown_timer[tid]["last"] = now

# reset 로직(ROI 안에 unknown이 더 이상 없으면 일정 시간 뒤 해제)
if any_unknown_inside:
    last_unknown_inside_ts = now
else:
    if alert_active and (now - last_unknown_inside_ts) >= RESET_SEC:
        alert_active = False
        unknown_timer.clear()

# 외부 연동을 위한 상태 출력/사용
value = 1 if alert_active else 0
# 예) if value == 1: send_kakao_alert(...)
```



---
### 자율주행 핵심 코드
- 소스코드 설명 :SLAM으로 /map 생성 → Nav2로 전역/지역 경로계획 → Waypoint Follower로 순차 주행 및 장애물 회피를 구현한 자율주행 코드입니다.

### Nav2 컴포저블 컨테이너 (Nav2 핵심 서버 호스팅)

역할 : Nav2의 서버들(플래너, 컨트롤러, BT 등)을 하나의 컨테이너에서 구동합니다.

```xml
<!-- [2] 컴포저블 컨테이너 -->
<node
  pkg="rclcpp_components"
  exec="component_container_isolated"
  name="$(var container_name)"
  output="screen"
  args="--ros-args --log-level $(var log_level)"
  if="$(var use_composition)"
>
  <param from="$(var params_file)"/>
  <param name="autostart" value="$(var autostart)"/>
  <param name="use_sim_time" value="$(var use_sim_time)"/>
  <remap from="/tf" to="tf"/>
  <remap from="/tf_static" to="tf_static"/>
</node>
```

### SLAM 실행 (Cartographer 노드 기동)

역할: LiDAR /scan을 받아 /map 프레임을 생성합니다.

```xml
<!-- [3] Cartographer SLAM -->
<arg name="cartographer_prefix" default="$(find-pkg-share pinky_cartographer)"/>
<arg name="cartographer_config_dir" default="$(var cartographer_prefix)/params"/>
<arg name="configuration_basename" default="nav2_cartographer_params.lua"/>
<arg name="lidar_topic" default="/scan"/>

<node
  pkg="cartographer_ros"
  exec="cartographer_node"
  name="cartographer_node"
  output="screen"
  args="-configuration_directory $(var cartographer_config_dir) -configuration_basename $(var configuration_basename)"
>
  <param name="use_sim_time" value="$(var use_sim_time)"/>
  <remap from="/scan" to="$(var lidar_topic)"/>
</node>
```

### 점유격자 맵 퍼블리시 (OccupancyGrid)

역할: SLAM 결과를 /map의 OccupancyGrid로 주기 퍼블리시합니다.

```xml
<!-- [4] Occupancy Grid 퍼블리시 -->
<arg name="occ_res" default="0.05"/>
<arg name="occ_period" default="1.0"/>

<node
  pkg="cartographer_ros"
  exec="cartographer_occupancy_grid_node"
  name="cartographer_occupancy_grid_node"
  output="screen"
  args="-resolution $(var occ_res) -publish_period_sec $(var occ_period)"
>
  <param name="use_sim_time" value="$(var use_sim_time)"/>
</node>
```

### Nav2 전체 포함 (웨이포인트 포함)

역할: 전역·지역 플래너, BT, 웨이포인트 팔로워 등을 한 번에 기동합니다.

```xml
<!-- [5] Nav2(웨이포인트 포함) 런치 포함 -->
<include file="$(find-pkg-share pinky_navigation)/launch/navigation_launch.xml">
  <arg name="params_file" value="$(var params_file)"/>
  <arg name="use_sim_time" value="$(var use_sim_time)"/>
  <arg name="autostart" value="$(var autostart)"/>
  <arg name="use_composition" value="$(var use_composition)"/>
  <arg name="use_respawn" value="$(var use_respawn)"/>
  <arg name="container_name" value="$(var container_name)"/>
  <arg name="lifecycle_nodes" value="$(var lifecycle_nodes_nav)"/>
</include>
```


