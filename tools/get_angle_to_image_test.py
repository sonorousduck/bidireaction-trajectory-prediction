import time
import socket
import json
import struct
import pickle
import cv2
from object_tracking import *
plt.style.use('ggplot')
matplotlib.use('tkagg')


if __name__ == "__main__":
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    cfg.merge_from_file("./configs/bitrap_np_JAAD.yml")
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # Load in image and run yolo to grab bounding boxes
    cv_image = cv2.imread(f"data/JAAD/images/video_0005/00046.png")
    plt.imshow(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    plt.show()

    # Find angle of bounding box from center of image







