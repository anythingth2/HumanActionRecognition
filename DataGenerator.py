import cv2
import ujson
import glob
import numpy as np
import pandas as pd

videos_dataset_path = 'olympic dataset'
keypoints_dataset_path = 'output'
annotation_path = 'annotation.csv'

annotation = pd.read_csv(annotation_path)


def normalize_keypoints(keypoints):
    xs, ys = keypoints.T
    xs = (xs - min(xs[xs > 0]))/(max(xs)-min(xs[xs > 0]))
    ys = (ys - min(ys[ys > 0]))/(max(ys)-min(ys[ys > 0]))
    keypoints = np.array([xs, ys]).T
    keypoints[keypoints < 0] = -1
    return keypoints


class Sample:
    def __init__(self, name, image, keypoints, label):
        self.name = name
        self.image = image
        self._keypoints = keypoints
        self.label = label

    @property
    def keypoints(self):
        return self._keypoints[:, :2]

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


def load(number_action, load_image=False):
    actions = []
    for vid_path in glob.glob(f'{keypoints_dataset_path}/*/*')[:number_action]:
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
            label = annotation[np.bitwise_and(np.bitwise_and(
                annotation.class_name == dirs[1], annotation.video_name == dirs[2]), annotation.image_name == img_name)]['label'].values[0]
            people_count = len(keypoint_json['people'])
            if people_count == 1:
                pose_keypoint = keypoint_json['people'][0]['pose_keypoints_2d']
                pose_keypoint = np.array(pose_keypoint).reshape((-1, 3))

                samples.append(
                    Sample(f'{vid_name}/{img_name}', frame, pose_keypoint,label))

            elif people_count == 0:
                pose_keypoint = np.ones(
                    (75,), dtype='float32').reshape((-1, 3)) * -1
                samples.append(
                    Sample(f'{vid_name}/{img_name}', frame, pose_keypoint,label))
            else:
                is_single_people = True
                break
        if not is_single_people:
            actions.append(HumanAction(samples))
    return actions


actions = load(10)