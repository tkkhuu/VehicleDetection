import cv2
import glob
import time
import numpy as np
import matplotlib.image as mpimg

from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CarClassifier:
    def __init__(self):
        self._model = None

    def _get_hog_features(self, img, orient, pix_per_cell, cell_per_block):
        # Otherwise call with one output      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=False, feature_vector=True)
        return features

    def _get_spatial_features(self, img, size=(32,32)):
        features = cv2.resize(img, size).ravel() 
        # Return the feature vector
        return features

    def _get_color_hist_features(self, img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    def _convert_cspace(self, img, cspace):
        if cspace == 'RGB':
            feature_img = np.copy(img)
        elif cspace == 'HSV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        return feature_img
        
    def extract_features(self, img_paths,
                         use_hog_features=True, hog_cspace='YUV', hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channel='ALL',
                         use_spatial_features=False, spatial_cspace='RGB', spatial_size=(32, 32),
                         use_color_hist_features=False, color_hist_cspace='RGB', color_hist_nbins=32, color_hist_bins_range=(0, 256)):
        
        if not (use_hog_features or use_spatial_features or use_color_hist_features):
            raise ValueError('At least one type of feature must be set to True')

        features = []
        for img_path in img_paths:
            img = mpimg.imread(img_path)
            img = np.uint8((img / np.max(img)) * 255)
            
            feature_vector = np.array([])
            if use_hog_features:
                feature_img = self._convert_cspace(img, hog_cspace)

                if hog_channel == 'ALL':
                    hog_features = []
                    for channel in range(feature_img.shape[2]):
                        hog_features.append(self._get_hog_features(feature_img[:,:,channel], hog_orient, hog_pix_per_cell, hog_cell_per_block))
                    hog_features = np.ravel(hog_features)
                else:
                    hog_features = self._get_hog_features(feature_img[:,:,hog_channel], hog_orient, hog_pix_per_cell, hog_cell_per_block)
                
                feature_vector = np.concatenate((feature_vector, hog_features))
                

            if use_spatial_features:
                feature_img = self._convert_cspace(img, spatial_cspace)
                spatial_features = self._get_spatial_features(feature_img, size=spatial_size)
                feature_vector = np.concatenate((feature_vector, spatial_features))

            if use_color_hist_features:
                feature_img = self._convert_cspace(img, color_hist_cspace)
                hist_features = self._get_color_hist_features(feature_img, nbins=color_hist_nbins, bins_range=color_hist_bins_range)
                feature_vector = np.concatenate((feature_vector, hist_features))
            
            features.append(feature_vector)
            
        return features

    def train(self, X_train, y_train, classifier='SVM'):
        if classifier == 'SVM':
            self._model = LinearSVC()
            self._model.fit(X_train, y_train)

    def score(self, X_test, y_test):
        return self._model.score(X_test, y_test)


def main():

    print('Preparing car images ...')
    cars = []
    for car_img_file in glob.glob('./data/vehicles/GTI_MiddleClose/*.png'): cars.append(car_img_file)
    for car_img_file in glob.glob('./data/vehicles/GTI_Far/*.png'): cars.append(car_img_file)
    for car_img_file in glob.glob('./data/vehicles/GTI_Right/*.png'): cars.append(car_img_file)
    for car_img_file in glob.glob('./data/vehicles/GTI_Left/*.png'): cars.append(car_img_file)
    for car_img_file in glob.glob('./data/vehicles/KITTI_extracted/*.png'): cars.append(car_img_file)

    print('Preparing not car images ...')
    notcars = []
    for notcar_img_file in glob.glob('./data/non-vehicles/Extras/*.png'): notcars.append(notcar_img_file)
    for notcar_img_file in glob.glob('./data/non-vehicles/GTI/*.png'): notcars.append(notcar_img_file)

    print('Creating classifier ...')
    car_classifier = CarClassifier()

    print('Extracting car features ...')
    car_features = car_classifier.extract_features(cars, use_hog_features=True)
    
    print('Extracting not car features ...')
    notcar_features = car_classifier.extract_features(notcars, use_hog_features=True)

    print('Stacking data ...')
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    print('Shuffling data ...')
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    
    print('Scaling data ...')
    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Training data ...')
    t=time.time()
    car_classifier.train(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(car_classifier.score(X_test, y_test), 4))

if __name__ == '__main__': main()
    




                


