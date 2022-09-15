#! /bin/sh
# This is a placeholder script for doing inference that should be packaged up into a webserver 
python3.9 detect.py --bucket network-machine-learning-artifacts --weights yolo-test-exp/632020cce69929002c2a0a73/weights/best.pt --device cpu --source ~/Downloads --save-txt --name out