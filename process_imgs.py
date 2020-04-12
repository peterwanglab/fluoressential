import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from natsort import natsorted
from scipy.stats import sem, skew
from imageio import mimwrite
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from skimage import img_as_float, img_as_ubyte

from fluoressential.process import preprocess_img, segment_clusters, segment_object
from fluoressential.utils import (setup_dirs, list_img_files,
    approx_half_life, rescale, plot, custom_styles, custom_palette)


def test_preprocess_img(img_path):
    final, sig = preprocess_img(img_path)
    plt.imshow(final, cmap='viridis')
    plt.show()


def test_segment_object(img_path):
    thr, img = segment_object(img_path, clbr=True)
    plt.imshow(img, cmap='viridis')
    plt.contour(thr, linewidths=0.3, colors='w')
    plt.show()


def test_bilinus_segmentation(bilinus_img_path, nucleus_img_path):
    nucl, nucl_img = segment_object(nucleus_img_path, clbr=True, offset=-1)
    cell, cell_img = segment_object(bilinus_img_path)
    plt.imshow(cell_img, cmap='viridis')
    plt.contour(cell, linewidths=0.3, colors='w')
    plt.contour(nucl, linewidths=0.2, colors='r')
    plt.show()


def test_cad_segmentation(cad_img_path):
    cell, cell_img = segment_object(cad_img_path, offset=0)
    dots, dots_img = segment_clusters(cad_img_path, offset=0)
    dots = dots & cell
    anti = dots ^ cell
    plt.imshow(cell_img, cmap='viridis')
    plt.contour(anti, linewidths=0.2, colors='w')
    plt.contour(dots, linewidths=0.1, colors='r')
    plt.show()


def process_bilinus(nucleus_dir, bilinus_dir, results_dir):
    setup_dirs(os.path.join(results_dir, 'imgs'))
    t = []
    yc = []
    yn = []
    nucl_imgs = []
    cell_imgs = []
    nucleus_imgfs = list_img_files(nucleus_dir)
    bilinus_imgfs = list_img_files(bilinus_dir)
    n_imgs = min(len(nucleus_imgfs), len(bilinus_imgfs))
    for i, (nu_imgf, bi_imgf) in enumerate(tqdm(zip(nucleus_imgfs, bilinus_imgfs), total=n_imgs)):
        nucl, nucl_img = segment_object(nu_imgf, clbr=True, offset=-1)
        cell, cell_img = segment_object(bi_imgf)
        cell = np.logical_or(cell, nucl)
        cyto = np.logical_xor(cell, nucl)
        nroi = nucl*cell_img
        croi = cyto*cell_img
        nnz = nroi[np.nonzero(nroi)]
        cnz = croi[np.nonzero(croi)]
        nucl_ave_int = np.mean(nnz)
        cyto_ave_int = np.mean(cnz)
        fname = os.path.splitext(os.path.basename(bi_imgf))[0]
        t.append(np.float(fname))
        yn.append(nucl_ave_int)
        yc.append(cyto_ave_int)
        if i == 0:
            nucl_cmin = np.min(nucl_img)
            nucl_cmax = 1.1*np.max(nucl_img)
            cell_cmin = np.min(cell_img)
            cell_cmax = 1.1*np.max(cell_img)
        figh = nucl_img.shape[0]/100
        figw = nucl_img.shape[1]/100
        fig, ax = plt.subplots(figsize=(figw, figh))
        axim = ax.imshow(nucl_img, cmap='viridis')
        axim.set_clim(nucl_cmin, nucl_cmax)
        ax.grid(False)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        cmimg = np.array(fig.canvas.renderer._renderer)
        nucl_imgs.append(cmimg)
        plt.close(fig)
        figh = cell_img.shape[0]/100
        figw = cell_img.shape[1]/100
        fig, ax = plt.subplots(figsize=(figw, figh))
        axim = ax.imshow(cell_img, cmap='viridis')
        axim.set_clim(cell_cmin, cell_cmax)
        ax.grid(False)
        ax.axis('off')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        cmimg = np.array(fig.canvas.renderer._renderer)
        cell_imgs.append(cmimg)
        plt.close(fig)
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(cell_img, cmap='viridis')
            axim.set_clim(cell_cmin, cell_cmax)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(cyto, linewidths=1.0, colors='w')
                ax.contour(nucl, linewidths=0.2, colors='r')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f')
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    data = np.column_stack((t, yc, yn))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='t,yc,yn', comments='')
    y = np.column_stack((yc, yn))
    plot(t, y, xlabel='Time (s)', ylabel='Ave FL Intensity',
        labels=['Cytoplasm', 'Nucleus'], save_path=os.path.join(results_dir, 'plot.png'))
    y_rescaled = np.column_stack((rescale(yc), rescale(yn)))
    plot(t, y_rescaled, xlabel='Time (s)', ylabel='Ave FL Intensity (Rescaled [0, 1])',
        labels=['Cytoplasm', 'Nucleus'], save_path=os.path.join(results_dir, 'plot01.png'))
    mimwrite(os.path.join(results_dir, 'nucl.gif'), nucl_imgs, fps=10)
    mimwrite(os.path.join(results_dir, 'cell.gif'), cell_imgs, fps=10)


def process_cad(img_dir, results_dir):
    """Analyze cad dataset and generate figures."""
    setup_dirs(os.path.join(results_dir, 'imgs'))
    t = []
    ya = []
    yd = []
    imgs = []
    for i, imgf in enumerate(tqdm(list_img_files(img_dir))):
        fname = os.path.splitext(os.path.basename(imgf))[0]
        cell, img = segment_object(imgf, offset=0)
        dots, _ = segment_clusters(imgf, offset=0)
        dots = dots & cell
        anti = dots ^ cell
        croi = cell * img
        aroi = anti * img
        droi = dots * img
        cnz = croi[np.nonzero(croi)]
        anz = aroi[np.nonzero(aroi)]
        dnz = droi[np.nonzero(droi)]
        cell_area = len(cnz)/39.0625
        anti_area = len(anz)/39.0625
        dots_area = len(dnz)/39.0625
        if i == 0:
            cmin = np.min(img)
            cmax = 1.1*np.max(img)
            init_cell_area = cell_area
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            anti_int = np.nan_to_num(np.mean(anz))
            dots_int = np.nan_to_num(np.mean(dnz) - np.mean(anz)) * dots_area / init_cell_area
        t.append(np.float(fname))
        ya.append(anti_int)
        yd.append(dots_int)
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(img, cmap='viridis')
            axim.set_clim(cmin, cmax)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.contour(cell, linewidths=0.4, colors='w')
                ax.contour(dots, linewidths=0.2, colors='r')
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f')
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            imgs.append(np.array(fig.canvas.renderer._renderer))
            fig.savefig(os.path.join(results_dir, 'imgs', fname + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    data = np.column_stack((t, ya, yd))
    np.savetxt(os.path.join(results_dir, 'y.csv'),
        data, delimiter=',', header='t,ya,yd', comments='')
    y = np.column_stack((ya, yd))
    plot(t, y, xlabel='Time (s)', ylabel='Norm FL Intensity',
        labels=['Anti Region', 'Dots Region'], save_path=os.path.join(results_dir, 'plot.png'))
    y = np.column_stack((rescale(ya), rescale(yd)))
    plot(t, y, xlabel='Time (s)', ylabel='Norm FL Intensity',
        labels=['Anti Region', 'Dots Region'], save_path=os.path.join(results_dir, 'plot01.png'))
    mimwrite(os.path.join(results_dir, 'cell.gif'), imgs, fps=10)
    return t, ya, yd


def combine_cad(results_dir):
    with plt.style.context(('seaborn-whitegrid', custom_styles)):
        fig, ax = plt.subplots()
        combined_ya = pd.DataFrame()
        combined_yd = pd.DataFrame()
        for i, data_dir in enumerate(natsorted([x[1] for x in os.walk(results_dir)][0])):
            csv_fp = os.path.join(results_dir, data_dir, 'y.csv')
            data = pd.read_csv(csv_fp)
            t = data['t'].values
            ya = data['ya'].values
            yd = data['yd'].values
            yaf = interp1d(t, ya, fill_value='extrapolate')
            ydf = interp1d(t, yd, fill_value='extrapolate')
            t = np.arange(round(t[0]), round(t[-1]) + 1, 1)
            ya = pd.Series([yaf(ti) for ti in t], index=t, name=i)
            yd = pd.Series([ydf(ti) for ti in t], index=t, name=i)
            combined_ya = pd.concat([combined_ya, ya], axis=1)
            combined_yd = pd.concat([combined_yd, yd], axis=1)
            ax.plot(ya, color='#BBDEFB')
            ax.plot(yd, color='#ffcdd2')
        ya_ave = combined_ya.mean(axis=1).rename('ya_ave')
        ya_std = combined_ya.std(axis=1).rename('ya_std')
        ya_sem = combined_ya.sem(axis=1).rename('ya_sem')
        ya_ci = 1.96*ya_sem
        yd_ave = combined_yd.mean(axis=1).rename('yd_ave')
        yd_std = combined_yd.std(axis=1).rename('yd_std')
        yd_sem = combined_yd.sem(axis=1).rename('yd_sem')
        yd_ci = 1.96*yd_sem
        combined_data = pd.concat([ya_ave, ya_std, ya_sem, yd_ave, yd_std, yd_sem], axis=1)
        combined_data.to_csv(os.path.join(results_dir, 'combined.csv'))
        ax.plot(ya_ave, color='#1976D2', label='Anti Region Ave')
        ax.plot(yd_ave, color='#d32f2f', label='Dots Region Ave')
        ax.fill_between(t, (ya_ave - ya_ci), (ya_ave + ya_ci), color='#1976D2', alpha=.1)
        ax.fill_between(t, (yd_ave - yd_ci), (yd_ave + yd_ci), color='#d32f2f', alpha=.1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Norm FL Intensity')
        ax.legend(loc='best')
        fig.savefig(os.path.join(results_dir, 'combined.png'),
            dpi=100, bbox_inches='tight', transparent=False, pad_inches=0)
        plt.close(fig)


def process_FRET(donor_dir, fret_dir, results_dir):
    setup_dirs(os.path.join(results_dir, 'imgs', 'donor'))
    setup_dirs(os.path.join(results_dir, 'imgs', 'fret'))
    y0 = []
    y1 = []
    yn = []
    donor_imgfs = list_img_files(donor_dir)
    fret_imgfs = list_img_files(fret_dir)
    for i, (don_imgf, fre_imgf) in enumerate(tqdm(zip(donor_imgfs, fret_imgfs), total=len(donor_imgfs))):
        don_thr, don_img = segment_object(don_imgf)
        fre_thr, fre_img = segment_object(fre_imgf)
        don_roi = don_img * don_thr
        fre_roi = fre_img * fre_thr
        don_pix = don_roi[np.nonzero(don_roi)]
        fre_pix = fre_roi[np.nonzero(fre_roi)]
        don_ave_int = np.mean(don_pix)
        fre_ave_int = np.mean(fre_pix)
        y0.append(don_ave_int)
        y1.append(fre_ave_int)
        yn.append(fre_ave_int/don_ave_int)
        cmin = min(np.min(don_pix), np.min(fre_pix))
        cmax = 1.1*max(np.max(don_pix), np.max(fre_pix))
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(don_img, cmap='viridis')
            axim.set_clim(cmin, cmax)
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f')
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', 'donor', str(i) + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
        with plt.style.context(('seaborn-whitegrid', custom_styles)), sns.color_palette(custom_palette):
            fig, ax = plt.subplots(figsize=(10,8))
            axim = ax.imshow(fre_img, cmap='viridis')
            axim.set_clim(cmin, cmax)
            ax.grid(False)
            ax.axis('off')
            fig.tight_layout(pad=0)
            cb = fig.colorbar(axim, pad=0.01, format='%.4f')
            cb.outline.set_linewidth(0)
            fig.canvas.draw()
            fig.savefig(os.path.join(results_dir, 'imgs', 'fret', str(i) + '.png'),
                dpi=100, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)
    data = np.column_stack((y0, y1, yn))
    np.savetxt(os.path.join(results_dir, 'data.csv'),
        data, delimiter=',', header='d,f,f/d', comments='')


if __name__ == '__main__':
    # img_path = '/home/phuong/data/LINTAD/LINuS/bilinus/0/0.05.tiff'
    # test_preprocess_img(img_path)
    # test_segment_object(img_path)


    # bil_img_path = '/home/phuong/data/LINTAD/LINuS/bilinus/0/0.05.tiff'
    # nuc_img_path = '/home/phuong/data/LINTAD/LINuS/nucleus/0/0.05.tiff'
    # test_bilinus_segmentation(bil_img_path, nuc_img_path)

    # nuc_root_dir = '/home/phuong/data/LINTAD/LINuS/nucleus'
    # bil_root_dir = '/home/phuong/data/LINTAD/LINuS/bilinus'
    # bil_res_dir = '/home/phuong/data/LINTAD/LINuS-results'
    # for img_dir in natsorted([x[1] for x in os.walk(nuc_root_dir)][0]):
    #     nucleus_dir = os.path.join(nuc_root_dir, img_dir)
    #     bilinus_dir = os.path.join(bil_root_dir, img_dir)
    #     out_dir = os.path.join(results_dir, img_dir)
    #     process_bilinus(nucleus_dir, bilinus_dir, out_dir)


    cad_img_path = '/home/phuong/data/LINTAD/CAD/0/66.65.tiff'
    test_cad_segmentation(cad_img_path)

    # root_dir = '/home/phuong/data/LINTAD/CAD/'
    # results_dir = '/home/phuong/data/LINTAD/CAD-results'
    # for img_dirname in natsorted([x[1] for x in os.walk(root_dir)][0]):
    #     img_dir = os.path.join(root_dir, img_dirname)
    #     process_cad(img_dir, results_dir)
    # combine_cad(results_dir)


    # don_dir = '/home/phuong/data/reed/dishB_YPet/yfp/'
    # fre_dir = '/home/phuong/data/reed/dishB_YPet/fret/'
    # res_dir = '/home/phuong/data/reed/dishB_YPet-results'
    # process_FRET(donor_dir=don_dir, fret_dir=fre_dir, results_dir=res_dir)
