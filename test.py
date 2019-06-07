from keras.models import load_model
import DataGenerator
import Visualize
import numpy as np
import cv2

model = load_model('model.h5')
x_test,y_test = DataGenerator.load_dataset(1,number_sample=100)
y_pred = model.predict(x_test,)

for i in range(len(x_test)):
    x = x_test[i]
    y = y_pred[i]
    y = np.argmax(y)

    print(y)
    
    image = Visualize.visualize(x[0])
    cv2.imshow('skeleton',image)
    cv2.waitKey(10)
    cv2.destroyAllWindows()