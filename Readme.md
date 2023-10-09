# Behavioural Cloning

In this project, an end-to-end deep learning using convolutional neural networks (CNNs) has been utilized to map the raw pixels from 3 front-facing cameras to the steering commands for a self-driving car. A simulator was used to capture the images during training laps around a track with various turns, curb styles, heights and pavement.

## Dependencies
- [Python](https://www.python.org)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Simulator](https://github.com/udacity/self-driving-car-sim)

## Data Collection
Since this project is based on Behavioural Cloning, training data had to be generated, by controlling the car in the simulator. Using the mouse and keyboard commands, the car was driven around the course, in the simulator. The cameras in front of the cameras capture photos continuously. For each frame of the image, the feature or image was stored in a file & the corresponding label measurement for the steering angle was also captured via a driving log (`.csv`) file.

## Data Preprocessing
Each image was normalized and mean centered to improve numerical stability,removing bias in the images and reduce overfitting. Additionally, using a `Cropping2D Layer` I was able to crop the images 75 pixels from the top of each frame and 20 pixels from the bottom. This was done to increase the performance time for the model to focus only on the areas that require training for the steering angle. 

## Architecture
Several networks such as LeNet and AlexNet were implemented and tested, but the best performance was provided by NVIDIA's autonomous drving team's network.

## Testing the model
To test your model, run : `python network_file-name.py model.h5`,
where `network_file_name` is the name of the python file where the network is defined . An `.h5` file easily stores the weights and model configuration in a single file. `model.h5` can be generated using Keras.


