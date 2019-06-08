import cv2
import ujson
import glob
import numpy as np
import pandas as pd
import random
from functools import reduce
from keras.utils import to_categorical

videos_dataset_path = 'olympic dataset'
keypoints_dataset_path = 'output'
annotation_path = 'annotation.csv'
enabled_datasets = ('basketball_layup', 'clean_and_jerk', 'snatch')
categories = ('non action', 'clean_and_jerk', 'snatch')

category_count = len(categories)

train_path = 'train_test_split/train'
test_path = 'train_test_split/test'


# annotation = pd.read_csv(annotation_path)

def normalize_keypoints(keypoints):
    xs, ys = keypoints.T
    tmp_xs = xs[xs > 0]
    tmp_ys = ys[ys > 0]
    if len(tmp_xs) == 0:
        return keypoints

    xs = (xs - min(tmp_xs))/(max(xs)-min(tmp_xs))
    ys = (ys - min(tmp_ys))/(max(ys)-min(tmp_ys))
    keypoints = np.array([xs, ys]).T
    keypoints[keypoints < 0] = -1
    return keypoints


class Sample:
    def __init__(self, name, image, keypoints, label):
        self.name = name
        self.image = image
        self.raw_keypoints = keypoints[:, :2]
        self.keypoints = normalize_keypoints(keypoints[:, :2])
        self.label = label

    @property
    def skeleton_image(self):
        frame = np.copy(self.image)
        for point in self.keypoints:
            cv2.circle(frame, tuple(point.astype('int32')), 2, (0, 0, 255))
        return frame


class HumanAction:
    def __init__(self, samples):
        self.samples = samples

    def visualize(self):
        for sample in self.samples:
            cv2.imshow(sample.name, sample.skeleton_image)
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyWindow(sample.name)
                break
            cv2.destroyWindow(sample.name)

    @property
    def keypoints(self):
        output = []
        for sample in self.samples:
            output.append(sample.keypoints)
        return output

    @property
    def labels(self):
        output = []
        for sample in self.samples:
            output.append(sample.label)
        return output

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


'output/snatch/GLoKBmPQzDo_00196_00635/I00006_keypoints.json'


def load_actions(number_action, load_image=False, specific_keypoint_paths=None):
    actions = []
    if specific_keypoint_paths is None:
        vid_paths = glob.glob(f'{keypoints_dataset_path}/*/*')
        if not (number_action == -1 or number_action > len(vid_paths)):
            vid_paths = vid_paths[:number_action]
    else:
        # vid_paths = []
        # for path in specific_keypoint_paths:
        #     vid_paths.extend(glob.glob(f'{path}/*'))
        vid_paths = specific_keypoint_paths
    random.shuffle(vid_paths)

    for vid_path in vid_paths:
        keypoint_paths = glob.glob(
            f'{vid_path}/*.json', recursive=True)
        keypoint_paths.sort()
        is_single_people = False

        samples = []
        for keypoint_path in keypoint_paths:
            dirs = keypoint_path.split('/')
            vid_name = '/'.join(dirs[1:-1])
            img_name = dirs[-1].split('_')[0]
            if load_image:
                frame = cv2.imread(
                    f'{videos_dataset_path}/{vid_name}/{img_name}.jpg')
            else:
                frame = None

            with open(keypoint_path, 'r') as f:
                keypoint_json = ujson.load(f)

            # ignore multiple people pose

            if 'label' in keypoint_json:
                label = keypoint_json['label']
            else:
                label = 0
            people_count = len(keypoint_json['people'])
            if people_count == 1:
                pose_keypoint = keypoint_json['people'][0]['pose_keypoints_2d']
                pose_keypoint = np.array(pose_keypoint).reshape((-1, 3))

                samples.append(
                    Sample(f'{vid_name}/{img_name}', frame, pose_keypoint, label))

            elif people_count == 0:
                pose_keypoint = np.ones(
                    (75,), dtype='float32').reshape((-1, 3)) * -1
                samples.append(
                    Sample(f'{vid_name}/{img_name}', frame, pose_keypoint, label))
            else:
                is_single_people = True
                break
        if not is_single_people :
            actions.append(HumanAction(samples))
    return actions


def load_dataset(timestep_per_sample, stride=None, number_sample=100, load_image=False, specify_dataset_paths=None):
    if stride == None:
        stride = timestep_per_sample

    specify_keypoint_paths = None
    
    if specify_dataset_paths is not None:
        specify_keypoint_paths = []
        for dataset_name in enabled_datasets:
            with open(f'{specify_dataset_paths}/{dataset_name}.txt', 'r') as f:
                vid_names = f.read().splitlines()
                specify_keypoint_paths.extend(
                    [f'{keypoints_dataset_path}/{dataset_name}/{name}' for name in vid_names])
    
    actions = load_actions(number_sample, load_image=load_image,
                           specific_keypoint_paths=specify_keypoint_paths)

    random.shuffle(actions)
    xs = []
    ys = []
    for action in actions:
        for i in range(0, len(action.samples) - stride+1, stride):
            x = []
            y = []
            for sample in action.samples[i:i+timestep_per_sample]:
                x.append(sample.keypoints.flatten())
                y.append(sample.label)
            xs.append(x)
            ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)
    ys = to_categorical(ys, category_count,)
    return xs, ys, actions


def load_train_dataset(timestep_per_sample=1,load_image=False):
    return load_dataset(timestep_per_sample, specify_dataset_paths=train_path,load_image=load_image)


def load_test_dataset(timestep_per_sample=1,load_image =False):
    return load_dataset(timestep_per_sample, specify_dataset_paths=test_path,load_image=load_image)
