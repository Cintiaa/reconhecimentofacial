# reconhecimentofacial
Detecção e Reconhecimento Facial


TECNOLOGIAS UTILIZADAS

OpenCV 4.0.1

Python 3.6.7

pip 9.0.1

------------------***********------------------

.gitignore

.idea/
src/cascades/data
src/db/
src/classificador/

------------------***********------------------

Necessário a criação de um diretório chamado classificador para armazenar os classificadores
EigenFace, FisherFace e LBPHFace gerados para o treinamento.

Necessário a criação de uma pasta chamada db para armazenar a base de imagens


captura - Realiza a captura de imagens para a criação de uma base de dados

treinamento - Nessa classe é realizado o treinamento com a base de dados criada e geração de classificadores, 
utilizando os algoritmos Eingenface, FisherFace e LBPHFace com base nos haarcascades disponibilizados pelo OpenCV.


Com base nos classificadores gerados, os testes são realizados nas classes teste_eigenface, teste_fisherface, teste_lbph
