import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from gpiozero import RGBLED,Buzzer,Servo
from colorzero import Color
import numpy as np
import board
import busio as io
import adafruit_mlx90614
import time


buzzer = Buzzer(24)
led = RGBLED(22,23,27)
servo = Servo(17)



def sensor():
    i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
    mlx = adafruit_mlx90614.MLX90614(i2c)
    while True:
        ambientTemp = "{:.2f}".format(mlx.ambient_temperature)
        targetTemp = "{:.2f}".format(mlx.object_temperature)

        print("Ambient Temperature:", ambientTemp, "°C")
        print("Target Temperature:", targetTemp, "°C")
        time.sleep(3)
        return targetTemp

cascPath = "yuz_algila/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
model = load_model("maske_algila.model")

video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list = []
    preds = []
    for (x, y, w, h) in faces:
        face_frame = frame[y:y + h, x:x + w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        faces_list.append(face_frame)
        if len(faces_list) > 0:
            preds = model.predict(faces_list)
        for pred in preds:
            (mask, withoutMask) = pred
        if mask > withoutMask:
            label = "Maske takili"
            color = (0, 255, 0)
            a=float(sensor())
            if(a<=35):
                buzzer.off()
                led.color = Color('green')
                servo.max()
            else:
                label = "Atesiniz yuksek"
                color = (0, 0, 255)
                buzzer.on()
                led.color = Color('red')
                servo.min()
        else:
            label = "Maskeyi takiniz"
            color = (0, 0, 255)
            buzzer.on()
            led.color = Color('red')
            servo.min()

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()