# -*- coding: utf-8 -*-
"""
Готовит копии текстур MVTec AD для честного сравнения методов на ОДНИХ И ТЕХ ЖЕ файлах.

Для каждого сдвига условий (none / light / rot90) и числа эталонов k создаёт папку
    <out>/<сдвиг>_<k>shot/<категория>/
        train/good/          - первые k нормальных снимков (без сдвига: эталон снят в «правильных» условиях)
        test/<тип>/          - тестовые снимки со сдвигом условий
        ground_truth/<тип>/  - маски дефектов (поворачиваются вместе со снимком)
Все снимки сохраняются в разрешении size x size (по умолчанию 256), чтобы экономить место.
Эту папку понимают и eval_mvtec.py (с --shift none), и anomalib (PatchCore).

Пример:
  python3 make_shifted.py --root ~/mvtec_anomaly_detection --out ~/mvtec_shifted --shots 1 5
"""
import argparse, glob, os
import numpy as np
from PIL import Image


def shift_rgb(img, shift):
    a = np.asarray(img, np.float32) / 255.0
    if shift == 'light':
        h, w = a.shape[:2]
        yy, xx = np.indices((h, w))
        gain = (0.6 + 0.8 * xx / w)[..., None]
        add = (0.15 * yy / h)[..., None]
        a = np.clip(a * gain + add, 0, 1)
    elif shift == 'rot90':
        a = np.rot90(a)
    return Image.fromarray((255 * a).round().astype(np.uint8))


def shift_mask(m, shift):
    a = np.asarray(m)
    if shift == 'rot90':
        a = np.rot90(a)
    return Image.fromarray(np.ascontiguousarray(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cat', nargs='+', default=['carpet', 'grid', 'leather', 'tile', 'wood'])
    ap.add_argument('--shifts', nargs='+', default=['none', 'light', 'rot90'])
    ap.add_argument('--shots', nargs='+', type=int, default=[1, 5])
    ap.add_argument('--size', type=int, default=256)
    a = ap.parse_args()
    root, out, S = os.path.expanduser(a.root), os.path.expanduser(a.out), a.size
    for sh in a.shifts:
        for k in a.shots:
            for cat in a.cat:
                dst = os.path.join(out, f'{sh}_{k}shot', cat)
                train = sorted(glob.glob(os.path.join(root, cat, 'train', 'good', '*.png')))[:k]
                os.makedirs(os.path.join(dst, 'train', 'good'), exist_ok=True)
                for p in train:
                    Image.open(p).convert('RGB').resize((S, S), Image.BILINEAR).save(
                        os.path.join(dst, 'train', 'good', os.path.basename(p)))
                n = 0
                for d in sorted(os.listdir(os.path.join(root, cat, 'test'))):
                    os.makedirs(os.path.join(dst, 'test', d), exist_ok=True)
                    if d != 'good':
                        os.makedirs(os.path.join(dst, 'ground_truth', d), exist_ok=True)
                    for p in sorted(glob.glob(os.path.join(root, cat, 'test', d, '*.png'))):
                        im = Image.open(p).convert('RGB').resize((S, S), Image.BILINEAR)
                        shift_rgb(im, sh).save(os.path.join(dst, 'test', d, os.path.basename(p)))
                        if d != 'good':
                            name = os.path.splitext(os.path.basename(p))[0] + '_mask.png'
                            m = Image.open(os.path.join(root, cat, 'ground_truth', d, name)).convert('L').resize((S, S), Image.NEAREST)
                            shift_mask(m, sh).save(os.path.join(dst, 'ground_truth', d, name))
                        n += 1
                print(f'{sh}_{k}shot/{cat}: эталонов {len(train)}, тестовых {n}')


if __name__ == '__main__':
    main()
