import cv2
import face_recognition
import joblib
import os

path = "./images"
images = []
images_no = []
mylist = os.listdir(path)
encodeList = []

for img in mylist:
    curImg = cv2.imread(f"{path}/{img}")
    images.append(curImg)
    images_no.append(img.split()[0])


def find_encodings(images):

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


encodeListKnown = find_encodings(images)
joblib.dump(encodeListKnown, "encoding.sav")
joblib.dump(images_no, "image_name.sav")


