# surgery-bot

## How to clone:
- mkdir surgery_bot
- cd into workspace
- git clone git@github.com:leonardwei2023/surgery-bot.git src
- run catkin_make

## How to use:
roslaunch planning TBD

## Packages
### Vision:
roslaunch realsense2_camera rs_camera.launch -> to bringup the realsense camera
#### Topics:
/vision/closure (Publisher)
#### Services:
/vision/getEntryPoint (Server)

### Planning:
##### Services:
/planning/beginSuture
