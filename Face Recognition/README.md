#터미널 A
./cam_start.sh

#터미널 B
(1)데이터 베이스 만들기(먼저 해줘야함)
python3 build_gallery_db.py \
  --zoo /home/pinky/hailo_examples/models \
  --det scrfd_2.5g--640x640_quant_hailort_hailo8l_1 \
  --rec arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1 \
  --gallery /home/pinky/hailo_examples/assets/H_B_dataset \
  --db /home/pinky/hailo_examples/assets/.face_db

(2)실시간 인식 + OSD(On Screen Display)
python3 fifo_scrfd_face_match_osd.py \
  --zoo /home/pinky/hailo_examples/models \
  --det scrfd_2.5g--640x640_quant_hailort_hailo8l_1 \
  --rec arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1 \
  --fifo /tmp/cam.yuv \
  --db /home/pinky/hailo_examples/assets/.face_db \
  --thresh 0.40 --smooth 7 --det-every 1 --min-face 110 \
  --margin 15 --unknown-time 3.0 --reset-time 5.0
  
##인자별 설명
--zoo /home/pinky/hailo_examples/models
DeGirum 로컬 모델 패키지의 부모 폴더.
안에 아래 폴더명 그대로 있어야 함:

scrfd_2.5g--640x640_quant_hailort_hailo8l_1/

scrfd_500m--640x640_quant_hailort_hailo8l_1/ (교체 테스트용)

arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1/
체크: ls /home/pinky/hailo_examples/models/<모델폴더>/model.json

--det scrfd_2.5g--640x640_quant_hailort_hailo8l_1
얼굴 검출(SCRFD) 모델 이름(=폴더명).

2.5g: 정확도↑, 속도↓

500m: 속도↑, 정확도↓
필요 시 값만 scrfd_500m... 으로 바꿔 바로 A/B 테스트 가능.

--rec arcface_mobilefacenet--112x112_quant_hailort_hailo8l_1
임베딩(ArcFace-MobileFaceNet) 모델 이름(=폴더명). 112×112 정합 후 임베딩 추출.

--fifo /tmp/cam.yuv
터미널 A의 ./cam_start.sh가 쓰는 FIFO 경로.
이 스크립트는 FIFO에서 YUV420(I420) 프레임을 읽어서 처리한다.

cam_start.sh의 출력 경로/해상도와 반드시 일치해야 함.

파이프 확인: ls -l /tmp/cam.yuv (파일 타입이 p).

--db /home/pinky/hailo_examples/assets/.face_db
LanceDB 데이터베이스 디렉터리.

먼저 build_gallery_db.py로 face 테이블을 만들어둬야 함(byeongchan/hyunjun 5장씩 등록).

테이블명 바꾸려면 --table 옵션을 추가로 줄 수 있음(기본 face).

--thresh 0.45
라벨 판정 임계값(코사인 유사도 평균, 0~1).
TrackSmoother가 같은 라벨의 평균 유사도(avg)를 계산하고, avg < thresh이면 라벨을 unknown으로 바꾼다.

오인식(타인을 hyunjun/byeongchan으로 착각)이 있다 → 0.50~0.55로 올려 보수적으로.

너무 자주 unknown이 뜬다 → 0.40~0.35로 내려 관대하게.

--smooth 7
스무딩 창(프레임 수). 최근 7프레임의 동일 라벨 유사도를 평균내 안정화.

값↑: 라벨 깜박임↓(안정) / 반응속도↓

값↓: 반응속도↑ / 깜박임↑
5-9 범위 권장. FPS가 높으면 7-11도 괜찮다.

--min-face 80
최소 얼굴 크기(픽셀). 이보다 작은 박스는 무시.

멀리서 얼굴이 작게 잡힌다 → 64~72로 낮추기(인식률↑, 노이즈↑)

가까이서 얼굴이 크게 잡힌다 → 96~120으로 올리기(오탐↓)

--det-every 2
검출 간격(프레임 단위). 매 2프레임마다 SCRFD 검출을 돌리고, 나머지는 트래커로 추적.

30FPS 소스면 검출은 ≈15Hz.

속도가 부족하면 3~4로 늘려 추적 비중↑(CPU/GPU/NPU 부담↓, 드리프트↑).

정확도/박스 안정이 더 필요하면 1(매 프레임 검출).
