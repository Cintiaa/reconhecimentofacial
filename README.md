# reconhecimentofacial
Detecção e Reconhecimento Facial


Tecnologias Utilizadas

* OpenCV 4.0.1
* Python 3.6.7
* pip 9.0.1



No .gitignore está:

.idea/

src/db/

src/classificador/


Para armazenar os classificadores EigenFace, FisherFace e LBPHFaace gerados a partir do algoritmo de treinamento
é preciso criar um diretório chamado "classificador" no projeto.

Para realizar o teste com as imagens é necessário a criação de um diretório chamado db. O 
treinamento pode ser realizado utilizando a base de dados YaleFace disponível em 
http://vision.ucsd.edu/content/yale-face-database ou criar uma base de dados, separando algumas imagens para o 
treinamento e teste. Será preciso alterar no nome das pastas no projeto caso seja utilizado o db Yale.


A captura das imagens para treinamento é feita através do algoritmo "captura".

treinamento - Nessa classe é realizado o treinamento com a base de dados criada e geração de classificadores, 
utilizando os algoritmos Eingenface, FisherFace e LBPHFace com base nos haarcascades disponibilizados pelo OpenCV.


Com base nos classificadores gerados, os testes são realizados nas classes teste_eigenface, teste_fisherface, teste_lbph
e teste_camera_lbph
