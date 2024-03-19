Relatório do trabalho: Classificação com Histogramas - Processamento de Imagens
Aluno: Bruno Aquiles de Lima

O método para classificar as imagens foi o seguinte passo a passo:
    1- Ler a imagem como RGB.
    2- Calcular o histograma de cada canal e normalizá-los.
    3- Comparar os histogramas respectivos de cada canal usando o método escolhido.
    4- O score final de similaridade é a média dos scores das comparações
    dos histogramas de cada canal.
    5- A imagem é classificada de acordo com os K melhores scores finais.

Sobre os parâmetros:
    Os parâmetros foram escolhidos na tentativa e erro, com o objetivo de maximizar a acurácia
    em algum dos métodos.

    - O parâmetro K ficou em 2 (quando há empate, a menor classe é escolhida).
    - A quantidade de intervalos do histograma foi 64, sendo o parâmetro que 
    mais influenciou na acurácia.

    Esses valores podem ser encontrados no começo do arquivo "histograma" como constantes.

Linha de bash que ajudou nos testes: 
    for i in {1..5}; do; python3 histograma.py "$i" images | grep "Accuracy"; done

Sobre os resultados:
    Alguns comportamentos foram interessantes de observar:

    - O método de intersecção de histogramas foi o que obteve a maior acurácia (0.88).
    No entanto, caso os histogramas não estejam normalizados, ele é o método com a pior
    acurácia (0.44).

    - Era esperado que a obtenção do valor de similaridade usando a média dos scores de cada canal,
    fosse resultar em um valor igual ao score de similaridade obtido com a comparação do
    histograma em escala de cinza. Porém, a média dos scores de cada canal
    obteve uma acurácia significativamente maior.
