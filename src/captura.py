import cv2
from datetime import datetime
import numpy


classificador = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt2.xml')


camera = cv2.VideoCapture(1)
amostra = 1
numeroAmostras = 1
largura, altura = 200, 200
dt = datetime.now()
da = datetime.strftime(dt,"%d%m%m%H%M%S")


print('Por favor, se posicione em frente a câmera e aperte a letra Q')
while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faceDetectada = classificador.detectMultiScale(imagemCinza, scaleFactor=1.3, minSize=(100, 100))


    for(x, y, l, a) in faceDetectada:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.imwrite('../../resources/verificacao/imagem-' + str(amostra) + '-' + str(da) + '.jpg', imagemFace)
            ##print('[Foto '+ str(amostra) + ' capturada com sucesso]')
            amostra += 1

    cv2.imshow('Face', imagem)
    cv2.waitKey(1)

    if(amostra >= numeroAmostras + 1):
        break

camera.release()
cv2.destroyAllWindows()

print('Imagem capturada com sucesso!')
