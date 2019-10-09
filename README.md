
Detecção e Reconhecimento Facial utilizando Python e OpenCV

Tecnologias Utilizadas
* OpenCV 4.1.1
* Python 3.7.4
* pip 19.2.3


No .gitignore está:
.idea/
src/db/
src/classificador/



ARMAZENAMENTO DE IMAGENS
As imagens para a verificação e treinamento estão armazenadas no diretório resources/verificao e resources/uploads respectivamente.
Para tal é necessário realizar a criação de ambos diretórios.

CLASSIFICADOR
Os classificadores criados à partir do arquivo de treinamento estão armazenadas no diretório classificador que está no .gitignore do projeto.
Criar diretório chamado "classificador" para armazenar os classificadores.

INFORMAÇÕES SOBRE O SISTEMA
O arquivo "captura.py" realiza a captura de imagens faciais através da WebCam
O arquivo "treinamento.py" realiza o treinamento com as imagens armazenadas no diretório resources/uploads e gera os classificadores que realizarão o reconhecimento das imagens capturadas
No arquivo "reconhecimento.py" é executado o reconhecimento da imagem capturada no arquivo captura.py utilizando os classificadores gerados à partir do treinamento e também a atribuição
de presença ao aluno, caso ele seja reconhecido.

