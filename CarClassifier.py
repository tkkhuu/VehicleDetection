import cv2
import glob
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label


def window_search(img, 
                    x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(32, 32), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def find_cars(img, ystart, ystop, scale, svc, X_scaler, 
             use_hog_features=True, hog_cspace='RGB', hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channel='ALL',
             use_spatial_features=False, spatial_cspace='RGB', spatial_size=(32, 32),
             use_color_hist_features=False, color_hist_cspace='RGB', color_hist_nbins=32, color_hist_bins_range=(0, 256)):
    
    bbox_list = []
    draw_img = np.copy(img)
    img = np.uint8((img / np.max(img)) * 255)
    
    img_tosearch = img[ystart:ystop,:,:]
    
    if scale != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Define blocks and steps as above
    nxblocks = (img.shape[1] // hog_pix_per_cell) - hog_cell_per_block + 1
    nyblocks = (img.shape[0] // hog_pix_per_cell) - hog_cell_per_block + 1 
    #nfeat_per_block = hog_orient*hog_cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // hog_pix_per_cell) - hog_cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    if use_hog_features:
        hog_ctrans_tosearch = convert_cspace(img_tosearch, hog_cspace)
        hog = [get_hog_features(hog_ctrans_tosearch[:,:,0], hog_orient, hog_pix_per_cell, hog_cell_per_block, flatten=False),
               get_hog_features(hog_ctrans_tosearch[:,:,1], hog_orient, hog_pix_per_cell, hog_cell_per_block, flatten=False),
               get_hog_features(hog_ctrans_tosearch[:,:,2], hog_orient, hog_pix_per_cell, hog_cell_per_block, flatten=False)]

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            feature_vector = np.array([])
            if use_hog_features:
                if hog_channel == 'ALL':
                    hog_features = []
                    hog_features.append(hog[0][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
                    hog_features.append(hog[1][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
                    hog_features.append(hog[2][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel())
                    hog_features = np.ravel(hog_features)
                    
                else:
                    hog_features = hog[hog_channel][ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                feature_vector = np.hstack((feature_vector, hog_features))
                
            xleft = xpos*hog_pix_per_cell
            ytop = ypos*hog_pix_per_cell

            # Extract the image patch
            if use_spatial_features:
                spatial_ctrans_tosearch = convert_cspace(img_tosearch, spatial_cspace)
                if (spatial_ctrans_tosearch[ytop:ytop+window, xleft:xleft+window].shape[0] == 0 or spatial_ctrans_tosearch[ytop:ytop+window, xleft:xleft+window].shape[1] == 0):
                    continue
                subimg = cv2.resize(spatial_ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                spatial_features = get_spatial_features(subimg, size=spatial_size)
                feature_vector = np.hstack((feature_vector, spatial_features))
            
            # Get color features
            if use_color_hist_features:
                hist_ctrans_tosearch = convert_cspace(img_tosearch, color_hist_cspace)
                if (hist_ctrans_tosearch[ytop:ytop+window, xleft:xleft+window].shape[0] == 0 or hist_ctrans_tosearch[ytop:ytop+window, xleft:xleft+window].shape[1] == 0):
                    continue
                subimg = cv2.resize(hist_ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                hist_features = get_color_hist_features(subimg, nbins=color_hist_nbins)
                feature_vector = np.hstack((feature_vector, hist_features))

            # Scale features and make a prediction
            try:
                test_features = X_scaler.transform(feature_vector.reshape(1, -1))   
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    bbox = ( (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart) )
                    bbox_list.append(bbox)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
            except ValueError:
                continue
                
    return bbox_list

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def get_hog_features(img, orient, pix_per_cell, cell_per_block, flatten):
    # Otherwise call with one output      

    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                    cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                    transform_sqrt=True, 
                    visualise=False, feature_vector=flatten)
    return features

def get_spatial_features(img, size=(32,32)):
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

def get_color_hist_features(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def convert_cspace(img, cspace):
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
        
def extract_features(rgb_img,
                     use_hog_features=True, hog_cspace='RGB', hog_orient=9, hog_pix_per_cell=8, hog_cell_per_block=2, hog_channel='ALL',
                     use_spatial_features=False, spatial_cspace='RGB', spatial_size=(32, 32),
                     use_color_hist_features=False, color_hist_cspace='RGB', color_hist_nbins=32, color_hist_bins_range=(0, 256)):
    
    if not (use_hog_features or use_spatial_features or use_color_hist_features):
        raise ValueError('At least one type of feature must be set to True')

    rgb_img = np.uint8((rgb_img / np.max(rgb_img)) * 255)
    
    feature_vector = np.array([])
    if use_hog_features:
        feature_img = convert_cspace(rgb_img, hog_cspace)

        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_img.shape[2]):
                hog_features.append(get_hog_features(feature_img[:,:,channel], hog_orient, hog_pix_per_cell, hog_cell_per_block, flatten=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_img[:,:,hog_channel], hog_orient, hog_pix_per_cell, hog_cell_per_block, flatten=True)
        
        feature_vector = np.concatenate((feature_vector, hog_features))

    if use_spatial_features:
        feature_img = convert_cspace(rgb_img, spatial_cspace)
        spatial_features = get_spatial_features(feature_img, size=spatial_size)
        feature_vector = np.concatenate((feature_vector, spatial_features))

    if use_color_hist_features:
        feature_img = convert_cspace(rgb_img, color_hist_cspace)
        hist_features = get_color_hist_features(feature_img, nbins=color_hist_nbins, bins_range=color_hist_bins_range)
        feature_vector = np.concatenate((feature_vector, hist_features))
    
    return feature_vector

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

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
    svc = LinearSVC()

    print('Extracting car features ...')

    car_features = []
    for img_path in cars:
        img = mpimg.imread(img_path)
        car_features.append(extract_features(img, use_hog_features=True, use_color_hist_features=True, use_spatial_features=True))
    
    print('Extracting not car features ...')
    notcar_features = []
    for img_path in notcars:
        img = mpimg.imread(img_path)
        notcar_features.append(extract_features(img, use_hog_features=True, use_color_hist_features=True, use_spatial_features=True))

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
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    print('Reading test image')
    image = mpimg.imread('bbox-example-image.jpg')

    print('Find cars ...')
    
    bbox = []
    bbox += find_cars(image, 400, 656, 0.5, svc, X_scaler, use_hog_features=True, use_color_hist_features=True, use_spatial_features=True)                              
    bbox += find_cars(image, 400, 656, 1, svc, X_scaler, use_hog_features=True, use_color_hist_features=True, use_spatial_features=True)
    bbox += find_cars(image, 400, 656, 1.5, svc, X_scaler, use_hog_features=True, use_color_hist_features=True, use_spatial_features=True)
    bbox += find_cars(image, 400, 656, 2, svc, X_scaler, use_hog_features=True, use_color_hist_features=True, use_spatial_features=True)
    bbox += find_cars(image, 400, 656, 2.5, svc, X_scaler, use_hog_features=True, use_color_hist_features=True, use_spatial_features=True)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    heat = add_heat(heat, bbox)
    heat = apply_threshold(heat, 20)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
 
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()
    

if __name__ == '__main__': main()
    




                


