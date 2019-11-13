import cv2
import os
import numpy as np
from PIL import Image
import random


detectorFacial = cv2.CascadeClassifier('./cascades/data/haarcascade-frontalface-default.xml')
eigenface = cv2.face.EigenFaceRecognizer_create(80, 30000)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbphface = cv2.face.LBPHFaceRecognizer_create(2, 2, 5, 5, 50)
largura, altura = 200, 200

def getImageId():
    caminhos = [os.path.join('../../resources/essex/treinamento', f) for f in os.listdir('../../resources/essex/treinamento')]
    faces = []
    ids = []
    for c in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(c), cv2.COLOR_BGR2GRAY)
        faceDetectada = detectorFacial.detectMultiScale(imagemFace, minNeighbors=6, scaleFactor=1.5, minSize=(30, 30), maxSize=(200,200))

        for (x, y, l, a) in faceDetectada:
            imagemFace = cv2.resize(imagemFace[y:y + a, x:x + l], (largura, altura))

            id = int(os.path.split(c)[-1].split(".")[1])

            ids.append(id)
            faces.append(imagemFace)
            cv2.imshow("Face", imagemFace)
            cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImageId()

print('Realizando o treinamento do classificador!!')

#criando classificador EigenFace
eigenface.train(faces, ids)
eigenface.write('./classificador/EigenFace.yml')


#criando classificador FicherFace
fisherface.train(faces, ids)
fisherface.write('./classificador/FisherFace.yml')


#criando classificador LBPHFace
lbphface.train(faces, ids)
lbphface.write('./classificador/LBPHFace.yml')

print('Treinamento realizado')
