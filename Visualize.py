import cv2
import numpy as np


def visualize(keypoints, image=None, actual_class = '',pred_class=''):
    vertexs = ((0, 1), (0, 15), (0, 16), (0, 17),
               (0, 18), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (8, 12), (9, 10), (10, 11), (11, 22), (11, 24), (12, 13), (13, 14), (14, 19), (14, 21), (15, 17), (16, 18), (19, 20), (22, 23))
    keypoints = keypoints.reshape((25, 2))

    if image is None:
        height,  width = (128, 128)
        image = np.zeros((height, width))
    else:
        height, width = image.shape[:2]

    for vertex in vertexs:
        p1 = tuple((keypoints[vertex[0]]).astype('int'))
        p2 = tuple((keypoints[vertex[1]]).astype('int'))
        cv2.line(image, p1, p2, (255, 255, 255))
    for keypoint in keypoints:
        c = tuple((keypoint).astype('int'))
        cv2.circle(image, c, 2, (0, 0, 255), 2)
    
    cv2.putText(image,'actual '+ actual_class, (0, height-50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, 'predict '+pred_class, (0, height-1),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image
