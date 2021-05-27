#Gerekli Kütüphaneler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from google.colab import drive

#data çağırma
drive.mount('/content/drive')
base_dir = '/content/drive/MyDrive/data/'
# Öğrenme hızı, Epochs, batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32
DIRECTORY = os.path.join(base_dir, 'test')
CATEGORIES = ["Masked", "No_Mask"]

print("[Bilgi] Fotğraflar yükleniyor...")
data = []
labels = []
for category in CATEGORIES:
path = os.path.join(DIRECTORY, category)
for img in os.listdir(path):
img_path = os.path.join(path, img)
image = load_img(img_path, target_size=(224, 224))
image = img_to_array(image)
image = preprocess_input(image)
data.append(image)
labels.append(category)

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels,
test_size=0.20, stratify=labels, random_state=42)

# veri büyütmek için eğitim görüntü oluşturucusunu oluşturma
aug = ImageDataGenerator(
rotation_range=20,
zoom_range=0.15,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.15,
horizontal_flip=True,
fill_mode="nearest")

# MobileNetV2 ağını yükleme, "FC katman setleri"
baseModel = MobileNetV2(weights="imagenet", include_top=False,
input_tensor=Input(shape=(224, 224, 3)))

#temel modelin üstüne yerleştirilecek modelin başını inşa etme
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#Model, eğiteceğimiz gerçek model olacaktır
model = Model(inputs=baseModel.input, outputs=headModel)

#temel modeldeki tüm katmanlar üzerinde döngü oluştuema ve ilk eğitim sürecinde güncelle nmemeleri için onları dondurma
for layer in baseModel.layers:
layer.trainable = False

# Modeli derleme
print("[bilgi] model derleniyor...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
metrics=["accuracy"])

# Modeli eğitmek
print("[bilgi] model eğitiliyor...")
H = model.fit(
aug.flow(trainX, trainY, batch_size=BS),
steps_per_epoch=len(trainX) // BS,
validation_data=(testX, testY),
validation_steps=len(testX) // BS,
epochs=EPOCHS)

#Test
print("[bilgi] deeğerlendirme...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)

# Rapor
print(classification_report(testY.argmax(axis=1), predIdxs,
target_names=lb.classes_))

# Eğitimiz modeli kaydetme
print("[bilgi] maske_algila modeli kaydediliyor...")
model.save("maske_algila.model", save_format="h5")

# Eğitim kaybını ve doğruluğunu çizme
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

