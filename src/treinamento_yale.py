import cv2
import os
import numpy as np
from PIL import Image

eigenface = cv2.face.EigenFaceRecognizer_create(80, 30000)
fisherface = cv2.face.FisherFaceRecognizer_create(3, 2000)
lbphface = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 50)

def getImagemComId():
    caminhos = [os.path.join('../../resources/yalefaces/treinamento', f) for f in os.listdir('../../resources/yalefaces/treinamento')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = Image.open(caminhoImagem).convert('L') #converte para scala de cinza
        imagemNP = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
        ids.append(id)
        faces.append(imagemNP)

    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinamento...")
eigenface.train(faces, ids)
eigenface.write('./classificador/EigenFace.yml')


#criando classificador FicherFace
fisherface.train(faces, ids)
fisherface.write('./classificador/FisherFace.yml')


#criando classificador LBPHFace
lbphface.train(faces, ids)
lbphface.write('./classificador/LBPHFace.yml')

print("Treinamento realizado")

