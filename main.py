import cv2
import numpy as np
import face_recognition
import os

path = "./images"
names = {
    191891: "Ashwani",
    191907: "Joginder",
    191914: "Mohit",
    191941: "Sudhanshu",
    191947: "Vanish",
}
images = []
images_no = []
mylist = os.listdir(path)
encodeList = []

for img in mylist:
    curImg = cv2.imread(f"{path}/{img}")
    image_to_encode = cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(image_to_encode)[0]
    images_no.append(img.split()[0])
    encodeList.append(encode)
    with open("encoding.txt", "a") as encoding_file:
        encoding_file.write(f"{encode} \n")

# def find_encodings(images):
#
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#         with open("encoding.txt", "a") as encoding_file:
#             encoding_file.write(f"{encode} \n")
#
#     return encodeList


# encodeListKnown = find_encodings(images)
print(len(encodeList))
