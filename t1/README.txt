Relatório do trabalho: 
Classificação com Histogramas - Processamento de Imagens

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

    - O parâmetro K ficou em 3.
    - A quantidade de intervalos do histograma foi 160, sendo o parâmetro que 
    mais influenciou na acurácia.
    - O range de valores escolhidos foi [1, 254].

    Esses valores podem ser encontrados no começo do arquivo "histograma" como constantes.

Usar somente a metade superior da imagem ajudou a aumentar a acurácia.

Linha de bash que ajudou nos testes: 
    for i in {1..5}; do; python3 histograma "$i" images | grep "Accuracy"; done

Sobre os resultados:
    Alguns comportamentos foram interessantes de observar:

    - Os métodos de Correlação e Bhattacharyya não sofrem com a não normalização dos histogramas,
    os demais métodos tem quedas significativas de acurácia.

    - O método de Bhattacharyya foi o que obteve a maior acurácia (0.88).

    - Era esperado que a forma da obtenção do valor de similaridade (média dos valores de cada canal)
    fosse resultar em um valor igual ao obtido com a comparação do histograma em 
    escala de cinza. Porém, a média dos scores de cada canal obteve uma 
    acurácia significativamente maior:

    Forma do histograma | Dist. Eucl. | Correl. | Chi-Square | Intersection | Bhattacharyya
    ------------------- | ----------- | ------- | ---------- | ------------ | --------------
    Escala de cinza     | 0.8         | 0.8     | 0.56       | 0.72         | 0.76
    ------------------- | ----------- | ------- | ---------- | ------------ | --------------
    Média dos 3 canais  | 0.84        | 0.84    | 0.72       | 0.84         | 0.88

