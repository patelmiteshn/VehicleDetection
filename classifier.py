import numpy as np
import glob
import time
import os
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score

from feature_generation import *

def load_data():
    cwd = 'data/'
    vehicle_folders = [
        'vehicles/GTI_Far/',
        'vehicles/GTI_Left/',
        'vehicles/GTI_MiddleClose/',
        'vehicles/GTI_Right/',
        'vehicles/KITTI_extracted/'
      ]
    non_vehicle_folders = [
        'non-vehicles/Extras/',
        'non-vehicles/GTI/'
      ]

    data = {'features': [], 'labels': [], 'v_cnt': 0, 'nv_cnt': 0}

    for folder in vehicle_folders:
        for im_path in glob.glob(cwd + folder + '*.png'):
            data['features'].append(im_path)
            data['labels'].append(1)
            data['v_cnt'] += 1

    for folder in non_vehicle_folders:
        for im_path in glob.glob(cwd + folder + '*.png'):
            data['features'].append(im_path)
            data['labels'].append(0)
            data['nv_cnt'] += 1


    data['features'] = extract_features(data['features'])
#     temp = data['features']
#     temp = np.array(temp)
#     print(temp)
#     print('shape of feature extracted image', temp.shape)
    data['scaler'] = StandardScaler().fit(data['features'])
    data['features'] = data['scaler'].transform(data['features'])
    data['labels'] = np.array(data['labels'])
    data['labels'] = np.array(data['labels'])
#     print('data count for each class: ', data['v_cnt'], ' ', data['nv_cnt'])
#     with open('data/data.p', 'wb') as f:
#         pickle.dump(data, f)
    return data



def train():
    print('Loading the data...')
    t = time.clock()

    data = load_data()

    X_scaler = data['scaler']

    t = abs(time.clock() - t)
    print('Loaded %d car images and %d non-car images in %d seconds.'
          % (data['v_cnt'], data['nv_cnt'], t))

    features, labels = shuffle(data['features'], data['labels'])
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=1)
    print('Split the data into %d training and %d testing examples.' % (y_train.shape[0], y_test.shape[0]))
    del data

    model = LinearSVC()
    print('Training the model...')

    t = time.clock()
    model.fit(X_train, y_train)
    t = abs(time.clock() - t)

    print('Model accuracy: %0.4f' % model.score(X_test, y_test))
    
    # Save Model 
    with open('data/model.p', 'wb') as f:
        pickle.dump([model, X_scaler], f)

    # 10 Fold cross Validation
    scores = cross_val_score(model, features, labels, cv=10)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    return model, X_scaler

model, X_scaler = train()