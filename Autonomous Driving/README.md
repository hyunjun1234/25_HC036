터미널1:센서키기

ros2 launch pinky_bringup bringup.launch.xml


터미널2:맵불러오기


ros2 launch pinky_navigation pinky_slam_bringup.launch.xml


터미널3:rviz2키고 로봇조종 (nav2 goal) / 핑키 안에서 실행

export DISPLAY=:0
ros2 launch pinky_navigation nav2_view.launch.xml


터미널4:맵저장

ros2 run nav2_map_server map_saver_cli -f <map name>