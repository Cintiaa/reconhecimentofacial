import pyodbc
import cv2
import os
import numpy as np
from PIL import Image
import datetime
import shutil

#CRIA CONEXÃO COM BANCO DE DADOS
conn = pyodbc.connect('Driver={SQL Server}; Server=localhost; Database=reconhecimentoFacial; UID=sa; PWD=teste123; Trusted_Connection=yes; ')

cursor = conn.cursor()

cursor.execute('SELECT DISTINCT t.IdTurma, t.Sigla, a.IdAluno, a.RA, a.Nome as Aluno, i.name '
                   'FROM Alunos a JOIN ImagemFaces i on a.IdAluno = i.IdAluno '
                   'JOIN TurmaAlunos ta on a.IdAluno = ta.IdAluno '
                   'JOIN Turmas t on t.IdTurma = ta.IdTurma '
                   'WHERE t.IsDeleted = 0 AND a.IsDeleted = 0 AND i.IsDeleted = 0')



detectorFacial = cv2.CascadeClassifier('./cascades/data/haarcascade-frontalface-default.xml')

#RECONHECIMENTO EIGENFACE
reconhecedorFacial = cv2.face.EigenFaceRecognizer_create()
reconhecedorFacial.read('./classificador/EigenFace.yml')



totalAcertos = 0
percentualAcertos = 0.0
totalConfianca = 0.0
largura, altura = 200, 200

caminhos = [os.path.join('../../resources/verificacao', f) for f in os.listdir('../../resources/verificacao')]
for c in caminhos:
    imagemFace = Image.open(c).convert('L')
    imagemFaceNP = np.array(imagemFace, 'uint8')
    facesDetectadas = detectorFacial.detectMultiScale(imagemFaceNP)


    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (0, 0, 255), 2)
        imagemFace = cv2.resize(imagemFaceNP[y:y + a, x:x + l], (largura, altura))
        f = cv2.imshow('Faces', imagemFaceNP)

        cv2.waitKey(10)
        idprevisto, confianca = reconhecedorFacial.predict(imagemFaceNP)

        idatual = int(os.path.split(c)[-1].split("-")[1].replace("imagem", ""))

        file = str(os.path.split(c)[-1])

        file2 = str(os.path.split(c)[-1].split("-")[-1])


        diretorio = "C:\\Users\\cintia-nunes\\Desktop\\Projeto\\resources\\verificacao\\" + file
        newDiretorio = "C:\\Users\\cintia-nunes\\Desktop\\Projeto\\resources\\uploads\\"

        print(idatual, idprevisto)
        foto = cv2.imread(file)
        if idatual != idprevisto:
            for row in cursor:
                if idprevisto == row.IdAluno:
                    nome = row.Aluno
                    print("Aluno reconhecido como: {}, confirma?".format(str(nome)))
                    s = input('Insira sim ou não: ')
                    if s != 'sim':
                        print('Por favor, retire uma nova foto')
                        # REMOVE A IMAGEM NÃO RECONHECIDA DO DIRETORIO
                        os.remove(diretorio)
                        break
                    if s == 'sim':
                        id = row.IdAluno
                        turma = row.IdTurma

                        #ALTERA O NOME DA IMAGEM
                        file3 = 'imagem' + '-' + str(id) + '-' + file2

                        # INSERE A PRESENÇA AO ALUNO
                        cursor.execute('INSERT INTO Presencas(QtdPresenca, DtAula, IdAluno, IdTurma, IsDeleted, createdAt, updateDAt) VALUES (?, ?, ?, ?, ?, ?, ?)',
                            (2, datetime.datetime.now(), id, turma, 0, datetime.datetime.now(), datetime.datetime.now()))
                        cursor.commit()
                        print('Presença atribuída com sucesso!!')

                        #CONVERTE A IMAGEM PARA BINARIO
                        f = open(diretorio, 'rb')
                        hexdata = f.read()
                        f.close()

                        #INSERE A IMAGEM RECONHECIDA AO BANDO DE DADOS
                        cursor.execute('INSERT INTO ImagemFaces (type, name, data, IsDeleted, IdAluno, createdAt, updatedAt) values (?,?,?,?,?,?,?)',
                                       ('image/jpeg', file3, pyodbc.Binary(hexdata), 0, id, datetime.datetime.now(), datetime.datetime.now()))
                        cursor.commit()

                        #MOVE A IMAGEM CAPTURADA PARA A PASTA DE TREINAMENTO
                        shutil.copy2(diretorio, newDiretorio)
                        #RENOMEIA A IMAGEM CONCATENANDO O ID DO ALUNO RECONHECIDO
                        os.rename(newDiretorio+file, newDiretorio+file3)
                        #REMOVE A IMAGEM RECONHECIDA DO DIRETORIO
                        os.remove(diretorio)
                        break

            cursor.close()
            #del cursor
