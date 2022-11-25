import cv2
import joblib
import face_recognition
import numpy

names = {
    191891: "Ashwani",
    191907: "Joginder",
    191914: "Mohit",
    191941: "Sudhanshu",
    191947: "Vanish",
}

image_no = joblib.load("image_name.sav")
encodingsKnown = joblib.load("encoding.sav")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodingsKnown, encodeFace)
        # if(sum(matches) > 3 ) add more images and minimum 7 matches must be true.
        # if sum(matches) <= 3:
        #     continue
        faceDis = face_recognition.face_distance(encodingsKnown, encodeFace)
        print(faceDis)
        matchIndex = numpy.argmin(faceDis)
        if faceDis[matchIndex] > 0.5:
            continue
        if matches[matchIndex]:
            roll_no = int(image_no[matchIndex])
            name = names[roll_no].upper()
            print(name)
