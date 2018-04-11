# Vehicle Detection
This project builds a computer vision pipeline to detect and track vehicles driving on the highway.

## Data
The pipeline trains the classifier on car images (looking from behind) and non-car images. These are set of 64x64 images provided by the KITTI and GTI dataset. There were 8792 car images and 8968 non car images. I used 80% for training and the other 20% for testing.

Car image exmaple:
![alt text][car_image]

Non-car image example:
![alt text][noncar_image]

## Features Extraction
### Histogram of Oriented Gradients (HOG) features
HOG features were used as features to be trained on. I randomly selected images from the dataset and applied the HOG operation on these images to help me tune the HOG parameters.

![alt text][hog_viz]

I experiemented on a different combinations of parameters, the ranges I picked for each parameter are reasonable in terms of speed performance.

| Configuration | Color Space | Number of Orientation | Pixels per Cell | Cell per Block | Channel |
| --------------|-------------|-----------------------|-----------------|----------------|---------|
| 1             | RGB         | 9                     | 8               | 2              | ALL     |
| 2             | HLS         | 9                     | 8               | 2              | ALL     |
| 3             | HSV         | 9                     | 8               | 2              | ALL     |
| 4             | LUV         | 9                     | 8               | 2              | ALL     |
| 5             | YUV         | 9                     | 8               | 2              | ALL     |
| 6             | YCrCb       | 9                     | 8               | 2              | ALL     |
| 7             | RGB         | 9                     | 16              | 2              | ALL     |
| 8             | HLS         | 9                     | 16              | 2              | ALL     |
| 9             | HSV         | 9                     | 16              | 2              | ALL     |
| 10            | LUV         | 9                     | 16              | 2              | ALL     |
| 11            | YUV         | 9                     | 16              | 2              | ALL     |
| 12            | YCrCb       | 9                     | 16              | 2              | ALL     |
| 13            | RGB         | 11                    | 8               | 2              | ALL     |
| 14            | HLS         | 11                    | 8               | 2              | ALL     |
| 15            | HSV         | 11                    | 8               | 2              | ALL     |
| 16            | LUV         | 11                    | 8               | 2              | ALL     |
| 17            | YUV         | 11                    | 8               | 2              | ALL     |
| 18            | YCrCb       | 11                    | 8               | 2              | ALL     |
| 19            | RGB         | 11                    | 16              | 2              | ALL     |
| 20            | HLS         | 11                    | 16              | 2              | ALL     |
| 21            | HSV         | 11                    | 16              | 2              | ALL     |
| 22            | LUV         | 11                    | 16              | 2              | ALL     |
| 23            | YUV         | 11                    | 16              | 2              | ALL     |
| 24            | YCrCb       | 11                    | 16              | 2              | ALL     |

The combination that I found the most accurate in classifying test images as well as detecting in a scenario was configuration 17.
In the visualization, this configuration seems to sketch out the shape of a car and distinguish other noncar features.

![alt text][final_hog_config]

### Color Features

Cars have their own unique colors, therefore, in addition to HOG features, I created a histogram of values for each color channel and used it as a features in training.

![alt text][color_hist]

### Spatial Features
Another feature that might add some uniqueness to car images that I experimented was spatial features. This techniques basically resize the image, vectorize the image matrix and use that vector as a feature for training. The size I used was 32x32.

## Training
I used SVM as the classifier for detection. The features (HOG, histogram of colors, spatial) were extracted and organized into a vector for each image. These features were then normalized and fed into the classifier for training.

## Sliding Window Search
The window sliding technique was used to search for cars in an images. I searched only on the lower half of the image since this is where the road and the cars are. Taking advantage of the fact that the car size depends on where it is on the road, I created various scaled windows and searched with smaller windows on pixels closer to the middle in the y direction and larger windows on lower y-pixels.

![alt text][sliding_window]

[car_image]: https://raw.github.com/tkkhuu/VehicleDetection/master/readme_img/car_image.png "Car Image"
[noncar_image]: https://raw.github.com/tkkhuu/VehicleDetection/master/readme_img/noncar_image.png "Non Car Image"
[hog_viz]: https://raw.github.com/tkkhuu/VehicleDetection/master/readme_img/hog_viz.png "Visualizing HOG"
[final_hog_config]: https://raw.github.com/tkkhuu/VehicleDetection/master/readme_img/final_hog_config.png "Final HOG Config"
[color_hist]: https://raw.github.com/tkkhuu/VehicleDetection/master/readme_img/color_hist.png "Color Histogram"
[sliding_window]: https://raw.github.com/tkkhuu/VehicleDetection/master/readme_img/sliding_window.png "Sliding Window"