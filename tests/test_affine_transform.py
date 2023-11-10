from yolox.data.data_augment import random_affine
import cv2
import random
img_path="/home/shinohara/Documents/YOLOX/datasets/white_cane_detection/2nd/2nd_stage/20190909/2nd_augmenttation_2m_3m_2person/val/1_confroom_background_87_526_072_538_009_0_0.jpg"

bbox_x0y0=[int(178.0), int(152.0)]
bbox_x1y1=[int(178.0+154.0), int(152.0+94.0)]

before=cv2.imread(img_path)
before=cv2.rectangle(before, bbox_x0y0, bbox_x1y1, color=(255,0,0), thickness=4)
cv2.imwrite("/tmp/before.png", before)

rotate_prob=1
bboxes=[bbox]
image_t, bboxes = random_affine(before, bboxes, degrees=0.0, translate=0.0, scales=0.0, shear=0.0)

after=image_t
bbox_x0y0=[int(bboxes[0][0]["bbox"][0]),int(bboxes[0][0]["bbox"][1])]
bbox_x1y1=[int(bboxes[0][0]["bbox"][0]+bboxes[0][0]["bbox"][2]),int(bboxes[0][0]["bbox"][1]+bboxes[0][0]["bbox"][3])]
after=cv2.rectangle(after, bbox_x0y0, bbox_x1y1, color=(255,0,0), thickness=4)
cv2.imwrite("/tmp/after.png", after)