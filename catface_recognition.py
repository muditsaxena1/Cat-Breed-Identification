# Cat Face Recognition

# data preprocessing
# importing libraries
import cv2
import numpy as np
import pandas as pd

#importing dataset
dataset = pd.read_csv('./data/cat_data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
Y = np.int8(Y)

catnames = {0: 'Birman',
            1: 'British Shorthair',
            2: 'Norwegian Forest',
            3: 'Persian'} 

# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
cat_dataset = []
font = cv2.FONT_HERSHEY_SIMPLEX
file_name = 'CatDataset'
dataset_path = './data/dataset/'

load = input("Do you want to load the data?('y'/'n')")
if load == 'y':
    # TODO error handling
    cat_dataset = np.load(dataset_path + file_name + '.npy')

if load != 'y':
    for i in range(len(X_train)):
        img = cv2.imread('./data/' + X_train[i][0])
        if type(img) != np.ndarray:
            print('Could not read file ' + X_train[i][0])
            continue
        scaleFactor = 1.3
        minNeighbours = 5
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbours)
        #print(str(len(faces)) + ' ' + str(scaleFactor) + " " + str(minNeighbours))
        while len(faces) == 0:
            if scaleFactor > 1.05:
                scaleFactor -= 0.05
            elif minNeighbours > 2:
                minNeighbours -= 1
                scaleFactor = 1.3
            else:
                print("Count not detect anything in " + X_train[i][0])
                break
            faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbours)
            #print(str(len(faces)) + ' ' + str(scaleFactor) + " " + str(minNeighbours))
        
        #if len(faces) > 0:
            #print(str(scaleFactor) + " " + str(minNeighbours))
            
        for (x,y,w,h) in faces[:1]:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
            roi_color = img[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color, (100, 100))
            cat_dataset.append(np.append(roi_color.flatten(), Y_train[i]))
            cv2.putText(img, catnames[Y_train[i]],(0,20), font, 0.8,(0,0,255),2,cv2.LINE_AA)
            cv2.imshow('Training ' + X[i][0],img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
cat_dataset = np.asarray(cat_dataset)
if load == 'n':
    save = input("Do you want to save the data?('y'/'n')")
    if save == 'y':
        np.save(dataset_path + file_name, cat_dataset)
        print("Dataset saved at: {}".format(dataset_path + file_name + '.npy'))

def distance(v1, v2):
	# Eucledian distance
	# sub 2 ndarrays, squaring them, taking sum of elements and square rooting 'em
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]

Y_pred = []
for i in range(len(X_test)):
    img = cv2.imread('./data/' + X_test[i][0])
    if type(img) != np.ndarray:
        print('Could not read file ' + X_test[i][0])
        continue
    scaleFactor = 1.3
    minNeighbours = 5
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbours)
    #print(str(len(faces)) + ' ' + str(scaleFactor) + " " + str(minNeighbours))
    while len(faces) == 0:
        if scaleFactor > 1.05:
            scaleFactor -= 0.05
        elif minNeighbours > 2:
            minNeighbours -= 1
            scaleFactor = 1.3
        else:
            print("Count not detect anything in " + X_test[i][0])
            break
        faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbours)
        #print(str(len(faces)) + ' ' + str(scaleFactor) + " " + str(minNeighbours))
    
    #if len(faces) > 0:
        #print(str(scaleFactor) + " " + str(minNeighbours))
    for (x,y,w,h) in faces[:1]:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        roi_color = img[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (100, 100))
        
        # flatten() functions flattens the multi-D arrays into 1-D array
        out = knn(cat_dataset, roi_color.flatten())
        # Draw rectangle in the original image
        Y_pred.append(int(out))
        cv2.putText(img, catnames[Y_pred[i]],(x,y+h-10), font, 0.8,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(img, catnames[Y_test[i]],(0,20), font, 0.8,(0,0,255),2,cv2.LINE_AA)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        cv2.imshow('Test ' + X[i][0],img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
Y_pred = np.asarray(Y_pred)
Y_pred = np.int8(Y_pred)
accuracy = (Y_test == Y_pred).sum()/len(Y_test) * 100
print("Accuracy is {}%".format(accuracy))

"""
import matplotlib.pyplot as plt
plt.scatter([0,1,2,3], [62.5, 93.75, 87.5, 81.25], color = 'red')
plt.plot([0,1,2,3], [62.5, 93.75, 87.5, 81.25], color = 'blue')
plt.title('Accuracy vs Random State')
plt.xlabel('Random State')
plt.ylabel('Accuracy')
plt.plot([0,3], [81.25, 81.25], 'r--')
plt.show()
"""
