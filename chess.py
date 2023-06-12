from google.colab import drive
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle

drive.mount('/content/drive')

data_dir = '/content/drive/My Drive/dataset'

images = []
labels = []

for i, folder in enumerate(os.listdir(data_dir)):
    for filename in os.listdir(os.path.join(data_dir, folder)):
        img_path = os.path.join(data_dir, folder, filename)
        images.append(img_path)
        labels.append(i)

images = np.array(images)
labels = np.array(labels)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float") / 255.0
    return img

train_images = np.array([preprocess_image(img_path) for img_path in train_images])
test_images = np.array([preprocess_image(img_path) for img_path in test_images])

train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

clf = SVC(kernel='linear', C=1, random_state=42)

clf.fit(train_images, train_labels)

preds = clf.predict(test_images)

print(classification_report(test_labels, preds))

input_image_path = '/content/drive/My Drive/dataset/clear/part_58.png'
input_image = preprocess_image(input_image_path)

input_image = input_image.reshape(1, -1)

prediction = clf.predict(input_image)

print('Predicted class:', prediction[0])

class_name = os.listdir(data_dir)[prediction[0]]
print("Class name:", class_name)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)