from noise import sp_noise
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt


# Printa uma tabela com os resultados dos filtros no terminal
def draw_table(filters: dict, noise_levels: list, psnr_ceil: float) -> None:
    print('PSNR de cada filtro para diferentes níveis de ruído:\n')
    print(f'{"Noise level":<15}{"Média":<15}{"Gaussiano":<15}{"Mediana":<15}{"Stacking":<15}{"NLM":<15}{"Bilateral":<15}')
    for i, nl in enumerate(noise_levels):
        print(f'{nl:<15}{filters["cv_blur"][i]:<15.2f}{filters["cv_ga_blur"][i]:<15.2f}{filters["cv_me_blur"][i]:<15.2f}{filters["stacking"][i]:<15.2f}{filters["cv_nlm"][i]:<15.2f}{filters["cv_bf"][i]:<15.2f}')

    print(f'\nPSNR máximo: {psnr_ceil:.2f}')
    print('\nAverage values:')
    print(f'Média: {np.mean(filters["cv_blur"]):.2f}')
    print(f'Gaussiano: {np.mean(filters["cv_ga_blur"]):.2f}')
    print(f'Mediana: {np.mean(filters["cv_me_blur"]):.2f}')
    print(f'Stacking: {np.mean(filters["stacking"]):.2f}')
    print(f'NLM: {np.mean(filters["cv_nlm"]):.2f}')
    print(f'Bilateral: {np.mean(filters["cv_bf"]):.2f}')
    print('-' * 100)

# Retorna o stop distance para o método de stacking com base em diferentes 
# níveis de ruído em @noise_levels sobre a imagem @img_original.
# Salva um gráfico com os resultados se @save_plot for True.
def catch_sd_stacking(img_original, noise_levels: list, save_plot: bool) -> float:
    sd_values = np.arange(0.1, 0.005, -0.005)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 linha, 2 colunas

    psnr_stb = [[], []]

    for nl in noise_levels:
        psnrs = []
        qnt_imgs = []
        for sd in sd_values:
            psnr, qnt = stacking_filter(img_original, nl, sd)
            psnrs.append(psnr)
            qnt_imgs.append(qnt)

        axs[0].plot(sd_values, psnrs, label=f'Noise level: {nl}')
        axs[1].plot(sd_values, qnt_imgs, label=f'Noise level: {nl}')

        # Calcular as derivadas
        deriv_psnrs = np.gradient(psnrs, sd_values)
        # Guarda os valores dos pontos onde o PSNR estabiliza (derivada < 0.5)
        for i, deriv in enumerate(deriv_psnrs):
            if abs(deriv) < 0.5:
                psnr_stb[0].append(sd_values[i])
                psnr_stb[1].append(psnrs[i])

    if save_plot:
        axs[0].scatter(psnr_stb[0], psnr_stb[1], color='black', 
                       marker='o', label='PSNR stabilizing')
        axs[0].set_title('PSNR x Stop distance')
        axs[0].set_xlabel('Stop distance')
        axs[0].set_ylabel('PSNR')
        axs[0].invert_xaxis()
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

        axs[1].set_title('Number of images x Stop distance')
        axs[1].set_xlabel('Stop distance')
        axs[1].set_ylabel('Number of images')
        axs[1].invert_xaxis()

        plt.tight_layout()
        plt.savefig(f'stacking_test.png')

    return float(np.mean(psnr_stb[0]))

# Faz o stacking de imagens ruidosas usando a média dos pixels até que a
# diferença de PSNR entre duas iterações seja menor que @stop_distance 
# Retorna o PSNR final e a quantidade de imagens usadas.
def stacking_filter(img_original, noise_level: float, stop_distance: float) -> tuple[float, int]:
    previous_psnr = 0
    psnr = 0
    img_stack = []
    while(True):
        img_stack.append(sp_noise(img_original, noise_level))
        img = np.mean(img_stack, axis=0).astype(np.uint8)
        psnr = cv2.PSNR(img_original, img)
        if psnr - previous_psnr < stop_distance:
            return psnr, len(img_stack)
        previous_psnr = psnr

# Testa os filtros cv2.blur, cv2.GaussianBlur, cv2.medianBlur e o método de stacking
# para diferentes níveis de ruído em @noise_levels usando o kernel size @ks.
def test_filters(img_original, noise_levels: list, ks: int, sd_stacking: float) -> dict:
    filters = { 'cv_blur': [], 'cv_ga_blur': [], 
               'cv_me_blur': [], 'stacking': [],
               'cv_nlm': [], 'cv_bf': []}
    for nl in noise_levels:
        img_noised = sp_noise(img_original, nl)
        psnr_cv_blur = cv2.PSNR(img_original, cv2.blur(img_noised, (ks, ks)))
        psnr_cv_ga_blur = cv2.PSNR(img_original, cv2.GaussianBlur(img_noised, (ks, ks), 0))
        psnr_cv_me_blur = cv2.PSNR(img_original, cv2.medianBlur(img_noised, ks))
        psnr_nlm = cv2.PSNR(img_original, cv2.fastNlMeansDenoising(img_noised, None, h=21))
        psnr_bf = cv2.PSNR(img_original, cv2.bilateralFilter(img_noised, ks, 10, 10))
        psrn_s, qnt_imgs = stacking_filter(img_original, nl, sd_stacking)
        # print(f'Noise level: {nl} - Qnt imgs: {qnt_imgs}')

        filters['cv_blur'].append(psnr_cv_blur)
        filters['cv_ga_blur'].append(psnr_cv_ga_blur)
        filters['cv_me_blur'].append(psnr_cv_me_blur)
        filters['stacking'].append(psrn_s)
        filters['cv_nlm'].append(psnr_nlm)
        filters['cv_bf'].append(psnr_bf)

    return filters

def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python3 testa_filtros.py <input1> <input2> ...')
        sys.exit(1)

    noise_levels = [0.01, 0.02, 0.05, 0.07, 0.1]

    for img_path in sys.argv[1:]:
        img_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            print(f'Could not open {img_path}')
            continue

        psnr_ceil = cv2.PSNR(img_original, img_original)
        # 0.047 (função bem demorada, então já deixei o valor fixo)
        # sd_stacking = catch_sd_stacking(img_original, noise_levels, True)
        sd_stacking = 0.047
        filters = test_filters(img_original, noise_levels, 3, sd_stacking)
        print(f'\nImage: {img_path}\n')
        draw_table(filters, noise_levels, psnr_ceil)


if __name__ == '__main__':
    main()
