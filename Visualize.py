import cv2
import numpy as np


def visualize( keypoints):
    vertexs = ((0, 1), (0, 15), (0, 16), (0, 17),
               (0, 18), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (8, 12), (9, 10), (10, 11), (11, 22), (11, 24), (12, 13), (13, 14), (14, 19), (14, 21), (15, 17), (16, 18), (19, 20), (22, 23))
    keypoints = keypoints.reshape((25, 2))
    width, height = (128, 128)
    image = np.zeros((height, width))
    for vertex in vertexs:
        p1 = tuple((keypoints[vertex[0]] * (width, height)).astype('int'))
        p2 = tuple((keypoints[vertex[1]] * (width, height)).astype('int'))
        cv2.line(image, p1, p2, (255, 255, 255))
    for keypoint in keypoints:
        c = tuple((keypoint*np.array([width, height])).astype('int'))
        cv2.circle(image, c, 2, (0, 0, 255), 2)
    return image
    