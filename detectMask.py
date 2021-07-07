import cv2 as cv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import time
import statistics

tiempos = [];

with_mask = np.load('withmask.npy')
without_mask = np.load('withoutmask.npy')
bad_mask = np.load('badmask.npy')

without_mask = without_mask.reshape(without_mask.shape[0], 50*50*3)
with_mask = with_mask.reshape(with_mask.shape[0],50 * 50 * 3)
bad_mask = bad_mask.reshape(bad_mask.shape[0],50 * 50 * 3)

x = np.r_[with_mask, without_mask]

labels = np.zeros(x.shape[0])
labels[with_mask.shape[0]:] = 1.0
labels[(without_mask.shape[0]+with_mask.shape[0]):] = 2.0
names = {0 : 'Mask', 1 : 'No Mask', 2 : 'Bad Mask'}


x_train, x_test, y_train, y_test = train_test_split(x, labels , test_size=0.25)

pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
svm = SVC()
svm.fit(x_train, y_train)
x_test  = pca.transform(x_test)
y_pred = svm.predict(x_test)

haar_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv.VideoCapture(0)
data = []
font = cv.FONT_HERSHEY_COMPLEX
while True:
    flag, img = capture.read()
    if flag:
        t0 = time.time();
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
            cv.putText(img,f'{w}x{h}',(x,y+h+8),cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,128),2)
            face = img[y:y+h, x:x+w, :]
            face = cv.resize(face, (50,50))
            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            
            cv.putText(img,n,(x,y),font,1,(244,250,250),2)
            #print(n)
            #print(int(pred))

        t1 = time.time();
        t = (t1-t0)*1000
        if (t > 0):
            tiempos.append(t)
        cv.putText(img,'{:.0f} ms'.format(t),(500,30),cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,128),2)
        cv.imshow('result',img)
        if cv.waitKey(2) == 27:
            break

capture.release()
cv.destroyAllWindows()

print(statistics.mean(tiempos))
