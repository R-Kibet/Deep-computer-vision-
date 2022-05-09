import os
import caer
import canaro
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import gc

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

img_size = (80,80)
channels = 1 #doesn't require any  colour code runs on  gray scale
char_path = "choose ur own path or liblary"

# loop through the lib and find how many images are there
char_dict = {}
for i in os.listdir(char_path):
    char_dict[i] = len(os.listdir(os.path.join(char_path, i)))

# sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)
char_dict

# grab the 1st 10 names and store them in a list
character = []  # empty list
count = 0
for i in char_dict:
    character.append(i[0])
    count += 1
    if count >= 10:
        break

print(character)


# create training data
train = caer.preprocess_from_dir(char_path ,character, channels = channels , IMG_SIZE = img_size, isShuffle = True)

length = len(train)
print(length)

# visualize the images present
plt.figure(figsize=(30,30))
plt.imshow(train[0][0], cmap = "gray")
plt.show()

# separate training set into features and labels
features , labels = caer.sep_train(train, IMG_SIZE = img_size)

# normalize the feature to be in training of (0,1)
features = caer.normalize(features)

"""
# labels is not a must to bel normalized but we one heart ecode them 
# turn them from numerical integers => binary class vectors
 done by tensorflow
"""
length2 = len(character)

labels = to_categorical(labels,length2)

# training and validation data ratio is always 80 train - 20 val
x_train, x_val, y_train, y_val = caer.train_val_split(features , labels, val_ratio=.2)

del train
del features
del labels
print(gc.collect())

batch_size = 32
Epoch = 10

# image data generator
# synthesis new images from existing images

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train , batch_size = batch_size)

# creating models
model = canaro.models.createSimpsonsModel(IMG_SIZE=img_size, channels=channels, output_dim=len(character),
                                         loss='binary_crossentropy', decay=1e-7, learning_rate=0.001, momentum=0.9,
                                         nesterov=True)
summ = model.summary()

print(summ)

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(train_gen,
                    steps_per_epoch=len(x_train)//batch_size,
                    epochs=Epoch,
                    validation_data=(x_val,y_val),
                    validation_steps=len(y_val)//batch_size,
                    callbacks = callbacks_list)

print(character)




