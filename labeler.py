import argparse
import cv2
import glob
import ujson
import os
import numpy as np
import csv
from functools import reduce

dataset_name = 'olympic dataset'
classes = ['clean_and_jerk', 'snatch']
obj = []
class_vids = []


def mark_label(vid_path):
    print('-'*20, '\n', vid_path, '\n')

    start_frame_index = None
    stop_frame_index = None

    imgs_path = glob.glob(f'{vid_path}/*')
    imgs_path.sort()
    seek = 0
    seek_keys = {
        'a': -10,
        's': -1,
        'd': 1,
        'f': 10,
        'n': -100,
        'm': 100
    }
    while seek < len(imgs_path):
        if seek < 0:
            seek = 0
        img_path = imgs_path[seek]

        img = cv2.imread(img_path)

        cv2.imshow(img_path, img)
        key = cv2.waitKey(0)
        key = chr(key)
        if key in seek_keys:
            seek_time = seek_keys[key]
            if seek + seek_time >= len(imgs_path):
                stop_frame_index = len(imgs_path) - 1
                print(f'mark stop frame [{seek}]')
                break
            seek += seek_time
            print(f'[{seek}] skip {seek_time}')
        elif key == 'k':
            if start_frame_index == None:
                start_frame_index = seek
                print(f'mark start frame [{seek}]')
            else:
                stop_frame_index = seek
                print(f'mark stop frame [{seek}]')
                break
        elif key == 'c':
            start_frame_index = None
            stop_frame_index = None
            print('clear!')
        elif key == 'q':
            return None

        cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    labels = list(map(lambda img_path: {
        'file_name': os.path.basename(img_path), 'label': 0}, imgs_path))
    for index in range(start_frame_index, stop_frame_index+1):
        labels[index]['label'] = 1
    return labels


for _class in classes:
    vid_infos = []
    class_paths = glob.glob(f'{dataset_name}/{_class}/*')
    for index, vid_path in enumerate(class_paths):

        print(f'{index}/{len(class_paths)}'.center(20, '-'))
        annotations = glob.glob(f'{vid_path}/*.json')
        if len(annotations) > 0:
            with open(annotations[0], 'r') as f:
                vid_info = ujson.load(f)
        else:
            vid_info = {'video_name': os.path.basename(vid_path)}
            vid_info['labels'] = mark_label(vid_path)

            with open(f'{vid_path}/annotation.json', 'w') as f:
                ujson.dump(vid_info, f)
        vid_infos.append(vid_info)
    obj.append({
        'class_name': _class,
        'annotation': vid_infos})

with open('annotation.json', 'w') as f:
    ujson.dump(obj, f)

with open('annotation.csv', 'w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['class_name', 'video_name','image_name', 'label'])

    def encode_csv_row(info):
        rows = []
        class_name = info['class_name']
        annotations = info['annotation']
        for anno in annotations:
            video_name = anno['video_name']
            for label in anno['labels']:
                rows.append([class_name, video_name, label['file_name'].split('.')[0],label['label']])
        return rows

    writer.writerows(reduce(lambda a,b:a+b,list(map(encode_csv_row, obj))))
# [
#     {
#         "class_name": "clean_and_jerk",
#         "annotation": [
#             {
#                 "video_name": "9XgsEBtBqm8_00001_00616",
#                 "labels": [
#                     {
#                         "file_name": "I00000.jpg",
#                         "label": 1
#                     },
