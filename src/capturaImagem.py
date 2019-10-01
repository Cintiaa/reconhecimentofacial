import cv2
import numpy


classificador = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt2.xml')
# classificadorOlho = cv2.CascadeClassifier('haarcascade-eye.xml')

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
amostra = 1
numeroAmostras = 1
largura, altura = 220, 220

print('Atenção se posicione em frente a câmera e em seguida pressione a tecla Q, para que possamos capturar uma imagem de sua face ')
print('Capturando as faces...')

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     scaleFactor=1.2,
                                                     minSize=(100, 100))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.imwrite('../../resources/imagefaces/treinamento/pessoa.' + str(amostra) +'.jpg', imagemFace)
            print('[Foto ' + str(amostra) + ' capturada com sucesso]')
            amostra += 1

    cv2.imshow('Face', imagem)
    cv2.waitKey(1)

    if(amostra >= numeroAmostras + 1):
        break

camera.release()
cv2.destroyAllWindows()