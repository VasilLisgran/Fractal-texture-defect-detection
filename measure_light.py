# -*- coding: utf-8 -*-
"""
Измеряет освещённость AITEX, чтобы проверить два утверждения статьи фактами:
  1) яркость различается между полосами одной ткани (разброс средних яркостей полос);
  2) яркость зависит от позиции вдоль полосы (виньетирование) - это смещение, о котором писал рецензент.
Края с фоном исключаются по тому же правилу, что в prepare_aitex_v3.py.
Для дефектных полос пиксели под маской не учитываются.

Пример:  python3 measure_light.py --root ~/aitex/archive
"""
import argparse, glob, os, re
from collections import defaultdict
import numpy as np
from PIL import Image

NAME = re.compile(r'^(\d{4})_(\d{3})_(\d{2})\.png$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--bg-frac', type=float, default=0.01)
    a = ap.parse_args()
    root, T = os.path.expanduser(a.root), a.tile
    normal, defect = defaultdict(list), defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, 'NODefect_images', '*', '*.png'))):
        m = NAME.match(os.path.basename(p))
        if m and not os.path.basename(p).startswith('._'):
            normal[m.group(3)].append(p)
    for p in sorted(glob.glob(os.path.join(root, 'Defect_images', '*.png'))):
        m = NAME.match(os.path.basename(p))
        if m and not os.path.basename(p).startswith('._'):
            defect[m.group(3)].append(p)

    print('ткань | полос норм./деф. | яркость полос: среднее, CV между полосами (норма), CV (дефектные) | '
          'профиль по позициям: размах средних яркостей (макс-мин)/среднее')
    for fab in sorted(normal):
        G = [np.asarray(Image.open(p).convert('L'), np.float64) for p in normal[fab]]
        n_pos = min(g.shape[1] for g in G) // T          # полосы бывают разной длины: берём общую часть
        fr = np.mean([[((g[:T, i*T:(i+1)*T] >= 250) | (g[:T, i*T:(i+1)*T] <= 5)).mean() for i in range(n_pos)] for g in G], 0)
        allowed = [i for i in range(n_pos) if fr[i] <= a.bg_frac]
        cols = np.concatenate([np.arange(i*T, (i+1)*T) for i in allowed])
        strip_means = np.array([g[:T, cols].mean() for g in G])
        pos_means = np.array([[g[:T, i*T:(i+1)*T].mean() for i in allowed] for g in G]).mean(0)
        dmeans = []
        for p in defect.get(fab, []):
            full = np.asarray(Image.open(p).convert('L'), np.float64)
            cd = cols[cols < full.shape[1]]                 # дефектная полоса может быть короче
            g = full[:T, cd]
            stem = os.path.basename(p)[:-4]
            ms = [np.asarray(Image.open(f).convert('L')) > 127 for f in glob.glob(os.path.join(root, 'Mask_images', stem + '_mask*.png'))
                  if not os.path.basename(f).startswith('._')]
            mk = np.logical_or.reduce(ms)[:T, cd] if ms else np.zeros_like(g, bool)
            if (~mk).sum():
                dmeans.append(g[~mk].mean())
        dmeans = np.array(dmeans)
        cv = strip_means.std() / strip_means.mean()
        cvd = dmeans.std() / dmeans.mean() if len(dmeans) > 1 else float('nan')
        prof = (pos_means.max() - pos_means.min()) / pos_means.mean()
        print(f'{fab}    | {len(G):3d}/{len(dmeans):3d}           | {strip_means.mean():6.1f}, CV {cv:.3f}, CV деф. {cvd:.3f}'
              f' | размах по позициям {prof:.3f} (позиции {allowed[0]}..{allowed[-1]})')
        print('       профиль яркости по позициям (норма): ' + ' '.join(f'{v:.0f}' for v in pos_means))


if __name__ == '__main__':
    main()