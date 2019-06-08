from keras.models import load_model
import DataGenerator
import Visualize
import numpy as np
import cv2
from functools import reduce

model = load_model('model.h5')
x_test, y_test, actions = DataGenerator.load_test_dataset(load_image=True)
len(actions)
samples = []
for action in actions:
    samples.extend(action.samples)

actions[0].samples[1].raw_keypoints

y_pred = model.predict(x_test,)
for i in range(len(x_test)):
    x = x_test[i]
    y = y_pred[i]
    y = np.argmax(y)
    label_pred = DataGenerator.categories[y]
    label_true = samples[i].name.split('/')[0]

    image = Visualize.visualize(
        samples[i].raw_keypoints, np.copy(samples[i].image), actual_class=label_true, pred_class=label_pred)
    cv2.imshow('skeleton', image)
    if cv2.waitKey(10) == ord('q'):
        break
cv2.destroyAllWindows()
