import cv2
import os
import numpy as np
from PIL import Image

detectorFace = cv2.CascadeClassifier('./cascades/data/haarcascade-frontalface-default.xml')
reconhecedorFace = cv2.face.EigenFaceRecognizer_create()
reconhecedorFace.read('./classificador/EigenFace.yml')

largura, altura = 200, 200

totalAcertos = 0
percentualAcertos = 0.0
totalConfianca = 0.0

caminhos = [os.path.join('../../resources/essex/teste', f) for f in os.listdir('../../resources/essex/teste')]
for caminhoImagem in caminhos:
    imagemFace = Image.open(caminhoImagem).convert('L')
    imagemFaceNP = np.array(imagemFace, 'uint8')
    facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (0, 0, 255), 2)
        imagemFace = cv2.resize(imagemFaceNP[y:y + a, x:x + l], (largura, altura))
        f = cv2.imshow('Faces', imagemFaceNP)

        cv2.waitKey(100)
        idprevisto, confianca = reconhecedorFace.predict(imagemFace)

        idatual = int(os.path.split(caminhoImagem)[-1].split(".")[0].replace("pessoa", ""))
        print(str(idatual) + " foi classificada como se fosse " + str(idprevisto) + " - " + str(confianca))

        if idprevisto == idatual:
            totalAcertos += 1
            totalConfianca += confianca

percentualAcertos = (totalAcertos / 10) * 100
totalConfianca = totalConfianca / totalAcertos

print("Percentual de acerto: " + str(percentualAcertos) + "%")
print("Total confiança: " + str(totalConfianca))

