sudo ip addr flush dev enp8s0  
添加自己的网段
sudo ip addr add 192.168.2.107/24 dev enp8s0  
查找雷达（最终结果为192.168.2.3）
for i in $(seq 1 254); do ping -c1 -W1 192.168.2.$i >/dev/null 2>&1 && echo ">>> 192.168.2.$i 在线" & done;wait


实验室标定版 
ros2 run camera_calibration cameracalibrator \
  --size 11x8 --square 0.15 \
  image:=/camera/camera/color/image_raw \
  camera:=/camera/camera/color
0.15m 11*8

ros2 run camera_calibration cameracalibrator \
--size 11x8 --square 0.15 \
--ros-args -r image:=/camera/camera/
color/image_raw \
-r camera:=/camera/camera/color \
-p image_format:=bgr8
