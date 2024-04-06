from noise import sp_noise
import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt


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

    axs[0].scatter(psnr_stb[0], psnr_stb[1], color='black', 
                   marker='o', label='PSNR stabilizing')

    if save_plot:
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

    return np.mean(psnr_stb[0])

# Faz o stacking de imagens ruidosas usando a média dos pixels até que a
# diferença de PSNR entre duas iterações seja menor que @stop_distance 
def stacking_filter(img_original, noise_level: float, stop_distance: float) -> float:
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
    for nl in noise_levels:
        img_noised = sp_noise(img_original, nl)
        psnr_cv_blur = cv2.PSNR(img_original, cv2.blur(img_noised, (ks, ks)))
        psnr_cv_ga_blur = cv2.PSNR(img_original, cv2.GaussianBlur(img_noised, (ks, ks), 0))
        psnr_cv_me_blur = cv2.PSNR(img_original, cv2.medianBlur(img_noised, ks))
        filters['cv_blur'].append(psnr_cv_blur)
        filters['cv_ga_blur'].append(psnr_cv_ga_blur)
        filters['cv_me_blur'].append(psnr_cv_me_blur)
        filters['stacking'].append(stacking_filter(img_original, nl, sd_stacking)[0])

    return filters

def main() -> None:
    noise_levels = [0.01, 0.02, 0.05, 0.07, 0.1]
    img_original = cv2.imread('original.jpg')
    psnr_ceil = cv2.PSNR(img_original, img_original)

    filters = { 'cv_blur': [], 'cv_ga_blur': [], 
               'cv_me_blur': [], 'stacking': [] }

    # 0.047
    # sd_stacking = catch_sd_stacking(img_original, noise_levels, True)
    sd_stacking = 0.047

    filters = test_filters(img_original, noise_levels, 3, sd_stacking)

    print('Noise levels:', noise_levels)
    print('cv_blur:', filters['cv_blur'])
    print('cv_ga_blur:', filters['cv_ga_blur'])
    print('cv_me_blur:', filters['cv_me_blur'])
    print('stacking:', filters['stacking'])


if __name__ == '__main__':
    main()
