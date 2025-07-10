import sys

sys.path.append(r'D:\work\ultralytics-main')

from ultralytics.data.converter import convert_dota_to_yolo_obb

convert_dota_to_yolo_obb(r'datadota2yoloobb')
#到ultralytics\data\converter.py下改map和图片格式