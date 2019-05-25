import cv2
import os
import numpy as np
from PIL import Image

detectorFacial = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt2.xml')
reconhecedorFacial = cv2.face.LBPHFaceRecognizer_create()
reconhecedorFacial.read('./classificador/LBPHFace.yml')


totalAcertos = 0
percentalAcertos = 0.0
totalConfianca = 0.0

caminhos = [os.path.join('./db/imagefaces/teste', f) for f in os.listdir('./db/imagefaces/teste')]
for caminhoImagem in caminhos:
    imagemFace = Image.open(caminhoImagem).convert('L')
    imagemFaceNP = np.array(imagemFace, 'uint8')
    facesDetectadas = detectorFacial.detectMultiScale(imagemFaceNP)

    for(x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (0, 0, 255), 2)
        cv2.imshow('Faces', imagemFaceNP)
        cv2.waitKey(10)
        idprevisto, confianca = reconhecedorFacial.predict(imagemFaceNP)
        idatual = int(os.path.split(caminhoImagem)[-1].split(".")[1].replace("pessoa", ""))
        print('{} foi classificado como se fosse {} - {}'.format(str(idatual), str(idprevisto), str(confianca)))

        if idprevisto == idatual:
            totalAcertos += 1
            totalConfianca += confianca

percentalAcertos = (totalAcertos / 5) * 100
totalConfianca = totalConfianca / totalAcertos

print('Percentual de acerto: ' + str(percentalAcertos) + '%')
print('Total de confiança: ' + str(confianca))