# OpenCV kütüphanesini dahil et 
import cv2

# İşletim sistemi işlemleri için os modülünü dahil et
import os

# NumPy kütüphanesini dahil et
import numpy as np

# Scikit-learn kütüphanesinden KMeans ve KNeighborsClassifier sınıflarını dahil et
from sklearn.cluster import KMeans # kmeans kümeleme algoritmaları için ilgili modülü içe aktarır
from sklearn.neighbors import KNeighborsClassifier # Knn en yakın komşu algoritması için ilgili modülü içe aktarır
import joblib # oluşturulan kmeans modelini dosyaya kaydetmek ve dosyadan okumak için kullanılır.
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Eğitim ve test veri seti dizinlerini belirle
train_path = "data/train/"
test_path = "data/test/"

# Eğitim veri seti sınıf adlarını tanımla
class_names_train = ["bedroom", "coast", "forest", "highway", "industrial", "insidecity", "kitchen", "livingroom", "mountain", "office", "opencountry", "store", "street", "suburb", "tallbuilding"]

# Test veri seti sınıf adlarını tanımla
class_names_test = ["bedroom", "coast", "forest", "highway", "industrial", "insidecity", "kitchen", "livingroom", "mountain", "office", "opencountry", "store", "street", "suburb", "tallbuilding"]

# Toplam sınıf sayısını belirle
num_classes = len(class_names_train)

# SIFT (Scale-Invariant Feature Transform) özellik çıkarıcıyı oluştur
sift = cv2.SIFT_create()

# Eğitim ve test sınıflarındaki SIFT özellik vektörlerini depolamak için kullanılır 
train_descriptors = []
test_descriptors = []

# Eğitim ve test etiketlerini depolamak için listeler oluştur
train_labels = []  
test_labels = []

# Her bir sınıf için eğitim ve test veri setlerini dolaş
for i in range(num_classes):
    # Eğitim veri seti için sınıf klasörünü belirle
    train_folder = os.path.join(train_path, class_names_train[i] + "/")
    
    # Eğitim veri setindeki dosyaları listele
    train_files = os.listdir(train_folder)
    
    # Her bir eğitim dosyasını işle
    for j in range(len(train_files)):
        # İlgili eğitim dosyasını oku
        img = cv2.imread(os.path.join(train_folder, train_files[j]))
        
        # Gri tonlamaya çevir ve boyutu değiştir
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))       

        
        # SIFT ile özellikleri çıkar. SIFT ile keypointler yani öznitelikler çıkartıldı.
        kp, des = sift.detectAndCompute(gray, None)
        
        # Özellikleri listeye ekle. Bu çıkarılın özellikler train_description'a eklenir
        train_descriptors.append(des)
        
        # Etiketi listeye ekle
        train_labels.append(i)

    # Test veri seti için sınıf klasörünü belirle
    test_folder = os.path.join(test_path, class_names_test[i] + "/")
    
    # Test veri setindeki dosyaları listele
    test_files = os.listdir(test_folder)
    
    # Her bir test dosyasını işle
    for j in range(len(test_files)):
        # İlgili test dosyasını oku
        img = cv2.imread(os.path.join(test_folder, test_files[j]))
        
        # Gri tonlamaya çevir ve boyutu değiştir
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        gray = cv2.resize(gray, (256, 256))
           
        
        # SIFT ile özellikleri çıkar. SIFT ile keypointler yani öznitelikler çıkartıldı.
        kp, des = sift.detectAndCompute(gray, None)
        
        # Özellikleri listeye ekle
        test_descriptors.append(des)
        
        # Etiketi listeye ekle
        test_labels.append(i)



# KMeans modelini oluştur
#kmeans = KMeans(n_clusters=100, n_init=5, random_state=0).fit(np.vstack(train_descriptors))
    #kmeans kümeleme modelini oluşturur. Eğitim veri kümesindeki tüm SIFT özelliklerini kullanarak KÜMELERİ bulur. 
#joblib.dump(kmeans,'kmeans_models_c100_i5_onisleme.joblib') 
    # elde edilen kümeleri bir dosyada depolar

kmeans = joblib.load('kmeans_models_c100_i5.joblib')
#kaydedilen kmeans kümesini yükle

#Kmans modelinden elde edilen küme merkezlerini alır. 
vocabulary = kmeans.cluster_centers_



##-------------
# Eğitim ve test veri kümesindeki SIFT ozellik vektörlerini kullanarak özellikleri çıkar.
# Eğtiim özelliklerini oluştur

#Bu kısım, eğitim ve test veri kümelerindeki SIFT özellik vektörlerini kullanarak özellik matrislerini oluşturuyor. Bu matrisler, her görüntüyü temsil etmek üzere kullanılacak özellikleri içerir.


train_features = np.zeros((len(train_descriptors), 100))
for i, descriptor in enumerate(train_descriptors):
    
    labels = kmeans.predict(descriptor)
    # bu aşamada eğitim SIFT vektörleriinin bulundukları dizideki elemanları kmeans modelini kullanrak tahmin yapar. 
    # Sıft özellik vektörünün hangi sınıfta veya kategoride olduğunu belirler
    # Kmeasn algıritması belli bir sayıda küme merkezi oluşturur ve her veri noktasını en yakın küme merkezine atar. 
    # bu predict metodu ile sıft özellik vektörlreini kümelere atar ve bu küme indexini labels değişleninde saklar
    
    for label in labels:
        train_features[i, label] += 1

# Test özelliklerini oluştur
test_features = np.zeros((len(test_descriptors), 100))
for i, descriptor in enumerate(test_descriptors):
    
    labels = kmeans.predict(descriptor)
    
    for label in labels:
        test_features[i, label] += 1
        
##--------------

# KNN sınıflandırma kısmı

# KNN modelini oluştur
knn = KNeighborsClassifier(n_neighbors=10)
# bir örneği sınflarndırmak için eğitim veri kümesindeki en yakın K komşuyu kullanır. her tahminde n komşu kullansın. 
# sınıflandırma modelini oluşturur ve eğitim veri kümesindeki özellik matrisleri ile etiketleri kullanarak modeli eğitir.
knn.fit(train_features, train_labels)
# Görüntüyü okuma
test_predictions = knn.predict(test_features)

# Her sınıf için doğruluk hesapla ve yazdır
for i in range(num_classes):
    class_indices = [j for j, label in enumerate(test_labels) if label == i]
    class_predictions = test_predictions[class_indices]
    class_true_labels = [i] * len(class_indices)
    
    class_accuracy = accuracy_score(class_true_labels, class_predictions)
    print(f"{class_names_test[i]} Sınıfı Doğruluğu: {class_accuracy}")

# Genel test doğruluğunu hesapla ve yazdır
overall_test_accuracy = accuracy_score(test_labels, test_predictions)
print("Genel Test Doğruluğu:", overall_test_accuracy)

#karmaşıklık matrisini yazdır

# Confusion Matrix'i hesapla
cm = confusion_matrix(test_labels, test_predictions)

# Seaborn kütüphanesi ile heatmap çizimi
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names_test, yticklabels=class_names_test)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Karmaşıklık Matrisi')
plt.show()
