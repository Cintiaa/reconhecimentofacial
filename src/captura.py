import cv2
import numpy


classificador = cv2.CascadeClassifier('./cascades/data/haarcascade_frontalface_alt2.xml')
# classificadorOlho = cv2.CascadeClassifier('haarcascade-eye.xml')

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
amostra = 1
numeroAmostras = 10
id = input('Digite seu identificador: ')
largura, altura = 220, 220
print('Capturando as faces...')

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     scaleFactor=1.2,
                                                     minSize=(100, 100))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        #regiao = imagem[y: y + a, x: x + l]
        #regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        #olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

        #for(ox, oy, ol, oa) in olhosDetectados:
            #cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            #if numpy.average(imagemCinza) > 110:
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
            cv2.imwrite('../../resources/imagefaces/treinamento/pessoa.' + str(id) + '.' + str(amostra) +'.jpg', imagemFace)
            print('[Foto ' + str(amostra) + ' capturada com sucesso]')
            amostra += 1

    cv2.imshow('Face', imagem)
    cv2.waitKey(1)

    if(amostra >= numeroAmostras + 1):
        break

camera.release()
cv2.destroyAllWindows()