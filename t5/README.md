# Dependências
PyTorch GPU - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`  
PyTorch CPU - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

# Como Rodar
Primeiramente, é necessário rodar o script get_macroscopic0_dataset.py, que irá baixar
o arquivo com as imagens e separá-las em 3 datasets, treino, validação e teste.

Após separar os datasets, basta rodar o script make_all_tests.py, que irá treinar 
e testar todos os métodos de extração de características e testa-los tanto com uma
rede neural específica (MLP) do método para classificação, quanto um KNN com k=1.
Baixar e separar o dataset:
```sh
python3 get_macroscopic0_dataset.py
```
Realizar o treinamento e testes:
```sh
python3 make_all_tests.py
```
Após a realização dos testes, é gerado uma tabela com a acurácia de cada método
usando o MLP e com o KNN no arquivo performance_table.csv.
