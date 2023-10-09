import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Conv2D,MaxPooling2D,Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy import ndimage
from math import ceil
import matplotlib.pyplot as plt

# Global Parameters
epochs = 5
batch_size = 10
validation_split = 0.2
correction = 0.2

lines = []
with open('/home/badri/Desktop/SDC/CarND-Behavioral-Cloning-P3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements =[]
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename= source_path.split('/')[-1]
                    curr_path = '/home/badri/Desktop/SDC/CarND-Behavioral-Cloning-P3/data/IMG/' + filename
                    image = cv2.imread(curr_path)
                    images.append(image)
    
                steering_center = float(line[3])
                steering_left = steering_center + 0.2
                steering_right = steering_center - 0.2

                measurements.extend([steering_center])
                measurements.extend([steering_left])
                measurements.extend([steering_right])

            X_train = np.array(images)
            #print(X_train.shape)
            Y_train = np.array(measurements)
            #print(Y_train.shape)

            # shuffle the data
            yield shuffle(X_train, Y_train)

# Utilize Generators
train_samples, validation_samples = train_test_split(lines, test_size=validation_split)
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    for gpu in gpus:
#        tf.config.experimental.set_memory_growth(gpu, True)

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape = (160,320,3))) # normalize the image by 255 the maximum for an image pixel between 0 and 1, then mean-center the image by subtracting -0.5 from each element, and give a image pixel of -0.5 to 0.5

model.add(Cropping2D(cropping=((75,25),(0,0)))) # cropping image 75 pixels from the top and 25 from the bottom, from "Even more powerful network video"

#NVIDIA END TO END NETWORK WITH MAX POOLING AND DROPOUTS ADDED AS DENOTED
model.add(Conv2D(24, (5,5), padding='valid', activation='relu')) # 24 filters 5x5 kernal
model.add(MaxPooling2D()) #ADDED
model.add(Dropout(0.5)) # ADDED dropout rate set to 0.5 for training/validation
model.add(Conv2D(36, (5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D()) #ADDED
model.add(Conv2D(48, (5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D()) #ADDED
model.add(Conv2D(64, (3,3), padding='valid', activation='relu'))
model.add(Conv2D(64, (1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D()) #ADDED
model.add(Dropout(0.5)) # dropout rate set to 0.5 for training/validation
model.add(Flatten())

# Next, four fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) # single output node = predicted steering angle


model.compile(loss='mse',optimizer='adam')
history_object = model.fit(train_generator, steps_per_epoch = ceil(len(train_samples)/batch_size), validation_data = validation_generator, validation_steps = ceil(len(validation_samples)/batch_size), epochs=epochs, verbose=1)
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### Keras outputs a history object that contains the training and validation loss for each epoch.
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
