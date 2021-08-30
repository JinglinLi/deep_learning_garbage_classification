from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from pprint import pprint
import cv2
import numpy as np


class PredictImage:
    """ 
    predict image class:
    input  : image captured from webcam
    output : MobileNet prediction : one of 1000 classes
             Garbage Class prediction : one of 12 classes
    """

    # garbage class
    CLASS_DICT = {'battery': 0,
    'biological': 1,
    'brown-glass': 2,
    'cardboard': 3,
    'clothes': 4,
    'green-glass': 5,
    'metal': 6,
    'paper': 7,
    'plastic': 8,
    'shoes': 9,
    'trash': 10,
    'white-glass': 11}

    # trained garbage prediction model : 
    # transfer learning model using 12-class-garbage data and pretrained MobileNet
    GM = load_model("garbage_model.h5")


    def __init__(self, image):
        self.image = image
        self.prediction_mn = []
        self.prediction_g = ()

    def preprocess_one_image(self):
        """image preprocessing for MobileNetV2"""
        im = self.image
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #plt.imshow(im)
        im_pp = image.img_to_array(im)
        im_pp = preprocess_input(im_pp)
        im_pp = im_pp.reshape(1, 224, 224, 3)
        return im_pp

    def predict_image_mobilenet(self):
        """predict one of 1000 classes in MobileNet"""
        im_pp = self.preprocess_one_image()
        m = MobileNetV2(weights='imagenet', include_top=True)
        # m.summary()
        prediction_mn = m.predict(im_pp)
        self.prediction_mn = decode_predictions(prediction_mn, 1)[0][0][1:]
        print('                                  ')
        print('MobileNet Prediction :', end=' ')
        pprint(self.prediction_mn)
        print('                                  ')

    def predict_image_garbage(self):
        """predict one of 12 garbage classes"""
        im_pp = self.preprocess_one_image()
        pred = PredictImage.GM.predict(im_pp) # probability for 12 classes
        class_ind = np.argmax(pred) # dictionary value for maximal probability
        class_probability = np.max(pred)
        class_name = [key for key, value in PredictImage.CLASS_DICT.items() if value == class_ind][0]
        self.prediction_g = (class_name, class_probability)
        print('                                  ')
        print(f'Garbage Class Prediction : {self.prediction_g}')
        print('                                  ')

        
        
        