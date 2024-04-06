- Kernel de tamanho 3. Outros tamanhos resultam em um PSNR menor para todos os
métodos.

- O método de stacking foi implementado da seguinte forma:
        É gerado uma imagem com ruído de valor N em cima da imagem original.
    Em seguida, é cálculado o PSNR da imagem ruídosa com a original. Se esse PSNR 
    tiver uma diferença com o PSNR anterior (que na primeira iteração é 0) 
    MAIOR que stop_distance (um valor passado por parâmetro), é gerada outra 
    imagem ruídosa com o mesmo valor N e é feito um stacking, cálculando a média 
    dos pixels de todas as imagens que estão no stack. Caso a diferença seja 
    MENOR o looping acaba.

        Para decidir o parâmetro stop_distance, foi feito um teste que envolve 
    pegar um valor de ruído e gerar PSNR's com stop_distance variando de 0.005 
    a 0.1 incrementando 0.005, em seguida, é anotado os pontos stop_distance x PSNR's
    que o PSNR estabiliza (derivada < 0.5. Valor escolhido através de testes).
    Esse mesmo processo ocorre para os outros valores de ruídos. No final, é tirado
    a média dos valores de stop_distance desses pontos onde o PSNR estabiliza,
    sendo esse o valor escolhido para o parâmetro (0.047).
