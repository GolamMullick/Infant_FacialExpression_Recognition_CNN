import tensorflow as tf

session_config = tf.ConfigProto()

session_config.gpu_options.allocator_type='BFC'

session_config.gpu_options.per_process_gpu_memory_fraction = 0.9

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('tf')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

PATH = os.getcwd()

data_path = PATH + '/data_3'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		input_img = cv2.imread(data_path + '/'+ dataset + '/'+ img )
		input_img_resize=cv2.resize(input_img,(224,224))
		img_data_list.append(input_img_resize)



print (img_data.shape)

num_classes = 5

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:1020]=0
labels[1020:2022]=1
labels[2022:2991]=2
labels[2991:3994]=3
labels[3994:]=4
	  
names = ['smile','cry','yaw','sleep','neutral']


X_train, X_test, y_train, y_test = train_test_split(img_data,Y, test_size=0.4, random_state=2)


image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')


model.summary()

last_layer = model.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
custom_vgg_model2.summary()

for layer in custom_vgg_model2.layers[:-3]:
    layer.trainable = False
    
custom_vgg_model2.summary()


custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

t=time.time()
#	t = now()
hist = custom_vgg_model2.fit(X_train, y_train, batch_size=32, epochs=42, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

score = custom_vgg_model2.evaluate(X_test, y_test,verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

print(hist.history.keys())

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

from sklearn.metrics import classification_report,confusion_matrix
import itertools
Y_pred = custom_vgg_model2.predict(X_test)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print(y_pred)
target_names = ['class 0(smile)', 'class 1(cry)', 'class 2(yaw)','class 3(sleep)','class 4(neutral)'] 4(neutral)']
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
pd.DataFrame(confusion_matrix(np.argmax(y_test,axis=1), y_pred),
             columns=['smile', 'cry', 'yaw', 'sleep', 'neutral'],
               index=['smile', 'cry', 'yaw', 'sleep', 'neutral']) 













