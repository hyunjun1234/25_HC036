#!/usr/bin/env bash
set -euo pipefail

FIFO=/tmp/cam.yuv
W=640
H=640
FPS=20

# FIFO 없으면 생성
[[ -p "$FIFO" ]] || mkfifo "$FIFO"

# libcamera-vid: 센서/노이즈/노출 고정으로 프레임 타이밍 안정화
libcamera-vid -t 0 --mode ${W}:${H} --width ${W} --height ${H} --vflip --framerate ${FPS} \
  --codec yuv420 --nopreview --libav-format rawvideo --ev 0.5 -o "$FIFO"
