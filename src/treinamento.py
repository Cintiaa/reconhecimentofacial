import cv2
import os
import numpy as np
from PIL import Image
import random


eigenface = cv2.face.EigenFaceRecognizer_create()

fisherface = cv2.face.FisherFaceRecognizer_create()

lbphface = cv2.face.LBPHFaceRecognizer_create()

def getImageComId():
    caminhos = [os.path.join('./db/imagefaces/treinamento', f) for f in os.listdir('./db/imagefaces/treinamento')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split(".")[1])
        print(id)
        ids.append(id)
        faces.append(imagemFace)
        cv2.imshow("Face", imagemFace)
        cv2.waitKey(10)
    return np.array(ids), faces
ids, faces = getImageComId()

print('Realizando Treinamento com as imagens!')

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
