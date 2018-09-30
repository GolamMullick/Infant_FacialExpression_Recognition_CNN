import tensorflow as tf
session_config = tf.ConfigProto()
session_config.gpu_options.allocator_type='BFC'
session_config.gpu_options.per_process_gpu_memory_fraction = 0.9

from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import time
from vgg16 import VGG16

img_width, img_height = 224, 224

model = load_model('model_feature_extraction.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']


img = image.load_img('test1.jpg', target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

Y_proba = model.predict(x)

y_classes = Y_proba.argmax(axis=-1)

print Y_proba

print y_classes

 

              
             
