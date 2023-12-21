import cv2
import os
import numpy as np
from sklearn.cluster import KMeans # kmeans kümeleme algoritmaları için ilgili modülü içe aktarır
from sklearn.neighbors import KNeighborsClassifier # Knn en yakın komşu algoritması için ilgili modülü içe aktarır
import joblib # oluşturulan kmeans modelini dosyaya kaydetmek ve dosyadan okumak için kullanılır.
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

train_path = "data/train/"
test_path = "data/test/"

class_names_train = ["bedroom", "coast", "forest", "highway", "industrial", "insidecity", "kitchen", "livingroom", "mountain", "office", "opencountry", "store", "street", "suburb", "tallbuilding"]

class_names_test = ["bedroom", "coast", "forest", "highway", "industrial", "insidecity", "kitchen", "livingroom", "mountain", "office", "opencountry", "store", "street", "suburb", "tallbuilding"]

num_classes = len(class_names_train)

sift = cv2.SIFT_create()

train_descriptors = []
test_descriptors = []

train_labels = []  
test_labels = []

for i in range(num_classes):
    train_folder = os.path.join(train_path, class_names_train[i] + "/")
    train_files = os.listdir(train_folder)
    for j in range(len(train_files)):
        img = cv2.imread(os.path.join(train_folder, train_files[j]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))       
        
        kp, des = sift.detectAndCompute(gray, None)
        train_descriptors.append(des)
        train_labels.append(i)

    test_folder = os.path.join(test_path, class_names_test[i] + "/")
    
    test_files = os.listdir(test_folder)
    
    for j in range(len(test_files)):
        img = cv2.imread(os.path.join(test_folder, test_files[j]))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        gray = cv2.resize(gray, (256, 256))
           
        
        kp, des = sift.detectAndCompute(gray, None)
        
        test_descriptors.append(des)
        
        test_labels.append(i)


kmeans = KMeans(n_clusters=100, n_init=5, random_state=0).fit(np.vstack(train_descriptors))
joblib.dump(kmeans,'kmeans_models_c100_i5_onisleme.joblib') 

kmeans = joblib.load('kmeans_models_c100_i5.joblib')
vocabulary = kmeans.cluster_centers_

train_features = np.zeros((len(train_descriptors), 100))
for i, descriptor in enumerate(train_descriptors):    
    labels = kmeans.predict(descriptor)
    for label in labels:
        train_features[i, label] += 1

test_features = np.zeros((len(test_descriptors), 100))
for i, descriptor in enumerate(test_descriptors):
    labels = kmeans.predict(descriptor)
    for label in labels:
        test_features[i, label] += 1
        
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_features, train_labels)
test_predictions = knn.predict(test_features)

for i in range(num_classes):
    class_indices = [j for j, label in enumerate(test_labels) if label == i]
    class_predictions = test_predictions[class_indices]
    class_true_labels = [i] * len(class_indices)
    
    class_accuracy = accuracy_score(class_true_labels, class_predictions)
    print(f"{class_names_test[i]} Sınıfı Doğruluğu: {class_accuracy}")

overall_test_accuracy = accuracy_score(test_labels, test_predictions)
print("Genel Test Doğruluğu:", overall_test_accuracy)

cm = confusion_matrix(test_labels, test_predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names_test, yticklabels=class_names_test)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Karmaşıklık Matrisi')
plt.show()
