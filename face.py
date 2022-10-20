# -*- coding: utf-8 -*-

import cv2
import numpy as np
import face_recognition
import os


i=0
def test_encode(encoded_face):
    global i
    face_index=encoded_face[i]
    i+=1
    return face_index
    
path = 'train'

images=[]
names=[]
lst=os.listdir(path)

for item in lst:
    Img= cv2.imread(f'{path}/{item}')
    images.append(Img)
    names.append(os.path.splitext(item)[0])

trainencoding_lst=[]

for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded_face = face_recognition.face_encodings(img,num_jitters=100,model='small')[0]
    trainencoding_lst.append(encoded_face)

# change the url
url = "https://192.168.0.108:8080/video"
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    test1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(test1)
    encoded_face = face_recognition.face_encodings(test1,faces_in_frame)
    
    i=0
    
    for face_loc in faces_in_frame:
        y1,x2,y2,x1 = face_loc
        count=-1
        face = test_encode(encoded_face)
        for j in trainencoding_lst:
            count +=1
            if face_recognition.compare_faces([j],face, tolerance=0.4) == [True]:
                cv2.rectangle(img, (x1,y1),(x2,y2), (0, 255, 0), 2)
                cv2.putText(img, names[count] , (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
                break
            else:
                cv2.rectangle(img, (x1,y1),(x2,y2), (0, 255, 0), 2)
                continue
    
    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
'''for i in range(len(faces)):
test_encode = face_recognition.face_encodings(test1)[i]
faces_in_frame = face_recognition.face_locations(test1)
face_loc=faces_in_frame[i]
y1,x2,y2,x1=face_loc
count=-1
for j in trainencoding_lst:
    count+=1
    if face_recognition.compare_faces([j], test_encode, tolerance=0.5) == [True]:
        
        cv2.rectangle(img, (x1,y1),(x2,y2), (0, 255, 0), 2)
        #cv2.rectangle(test, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
        cv2.putText(img, names[count] , (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)
        
        break
    else:
        cv2.rectangle(img, (x1,y1),(x2,y2), (0, 255, 0), 2)

cv2.imshow('video',img)
if cv2.waitKey(1) & 0xFF == ord('q'):
break

cap.release()
cv2.destroyAllWindows()'''






