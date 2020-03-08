import pandas as pd
import numpy as np
import os
import tensorflow.python.keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.inception_v3 import decode_predictions
from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate

# In[2]:


base_model=InceptionV3(weights='imagenet',include_top=False, input_shape= (224,224,3)) #imports the mobilenet model and discards the last 1000 neuron layer.

base_model.trainable = False

add_model = Sequential()
add_model.add(base_model)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.5))
add_model.add(Dense(9,
                    activation='softmax'))
model = add_model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4,
                                       momentum=0.9),
              metrics=['accuracy'])
model.summary()
'''
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(9,activation='softmax')(x) #final layer with softmax activation
'''
''
# In[3]:


'''model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


# In[4]:


# In[5]:

'''
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('./train/', # this is where you specify the path to the main data folder
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)


# In[33]:

'''
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

model.summary()
'''

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=1,
                   epochs=5)


img_path = 'test_kayak.jpg'
img = image.load_img(img_path, target_size=(224,224))
x= image.img_to_array(img)
x=np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model.predict(x)
print(features[0])
print(features)
plt.imshow(image.load_img(img_path))
plt.show()
