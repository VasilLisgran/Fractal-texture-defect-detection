# -*- coding: utf-8 -*-
"""
Готовит AITEX (версия с Kaggle: Defect_images / Mask_images / NODefect_images/<папка ткани>)
в структуре MVTec, чтобы eval_mvtec.py и run_patchcore.py работали без изменений.

Полосы ткани 4096 x 256 режутся на квадраты 256 x 256 в исходном разрешении (16 квадратов на полосу).
Для каждой ткани (fabric_00 ... fabric_06) и каждого числа эталонов k:
    <out>/<k>shot/fabric_XX/train/good/          - k квадратов из ПЕРВОЙ нормальной полосы (равномерно по длине)
    <out>/<k>shot/fabric_XX/test/good/           - квадраты из остальных нормальных полос (каждый 4-й, чтобы не раздувать тест)
    <out>/<k>shot/fabric_XX/test/defect/         - квадраты дефектных полос, в которых есть дефект по маске
    <out>/<k>shot/fabric_XX/ground_truth/defect/ - маски этих квадратов (объединение mask, mask1, mask2)
Квадраты дефектных полос БЕЗ дефекта по маске в тест не идут (во избежание неразмеченных дефектов).
Позиции на краю полосы, где в кадр попадает фон (а не ткань), исключаются целиком (и из эталонов, и из теста):
позиция исключается, если на нормальных полосах этой ткани в ней в среднем больше --bg-frac пикселей фона
(очень светлые >= 250 или очень тёмные <= 5). Список исключённых позиций печатается.
Эталонная полоса в тест не попадает.

Пример:
  python3 prepare_aitex.py --root ~/aitex/archive --out ~/aitex_prepared --shots 1 4
"""
import argparse, glob, os, re
from collections import defaultdict
import numpy as np
from PIL import Image

NAME = re.compile(r'^(\d{4})_(\d{3})_(\d{2})\.png$')   # снимок_дефект_ткань


def tiles(arr, T):
    n = arr.shape[1] // T
    return [arr[:T, i * T:(i + 1) * T] for i in range(n)]


def background_positions(strips, T, bg_frac):
    """Индексы квадратов, в которых на нормальных полосах заметная доля фона."""
    fr = None
    for p in strips:
        g = np.asarray(Image.open(p).convert('L'))
        t = [((x >= 250) | (x <= 5)).mean() for x in tiles(g, T)]
        fr = np.array(t) if fr is None else fr + np.array(t)
    fr = fr / len(strips)
    return [i for i, f in enumerate(fr) if f > bg_frac], fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='папка archive (в ней Defect_images, Mask_images, NODefect_images)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--shots', nargs='+', type=int, default=[1, 4])
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--good-every', type=int, default=4, help='из нормальных полос брать каждый N-й квадрат')
    ap.add_argument('--min-defect-px', type=int, default=20, help='квадрат считается дефектным, если в маске не меньше стольких пикселей')
    ap.add_argument('--bg-frac', type=float, default=0.01, help='порог доли пикселей фона для исключения позиции')
    ap.add_argument('--keep-edges', action='store_true', help='не исключать краевые позиции (старое поведение)')
    a = ap.parse_args()
    root, out, T = os.path.expanduser(a.root), os.path.expanduser(a.out), a.tile

    # нормальные полосы по тканям
    normal = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, 'NODefect_images', '*', '*.png'))):
        b = os.path.basename(p)
        m = NAME.match(b)
        if b.startswith('._') or not m:
            print('пропущен (нестандартное имя):', p)
            continue
        normal[m.group(3)].append(p)

    # дефектные полосы и их маски
    defect = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, 'Defect_images', '*.png'))):
        b = os.path.basename(p)
        m = NAME.match(b)
        if b.startswith('._') or not m:
            continue
        defect[m.group(3)].append(p)

    def load_mask(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        files = [f for f in glob.glob(os.path.join(root, 'Mask_images', stem + '_mask*.png')) if not os.path.basename(f).startswith('._')]
        if not files:
            return None
        ms = [np.asarray(Image.open(f).convert('L')) > 127 for f in files]
        return np.logical_or.reduce(ms)

    for k in a.shots:
        for fab in sorted(normal):
            dst = os.path.join(out, f'{k}shot', f'fabric_{fab}')
            for d in ('train/good', 'test/good', 'test/defect', 'ground_truth/defect'):
                os.makedirs(os.path.join(dst, d), exist_ok=True)
            strips = normal[fab]
            ref = np.asarray(Image.open(strips[0]).convert('RGB'))
            rt = tiles(ref, T)
            bad, fr = ([], None) if a.keep_edges else background_positions(strips, T, a.bg_frac)
            allowed = [i for i in range(len(rt)) if i not in bad]
            if bad:
                print(f'  fabric_{fab}: исключены позиции с фоном {bad} (доля фона: ' +
                      ', '.join(f'{i}:{fr[i]:.2f}' for i in bad) + ')')
            # эталоны: k квадратов, равномерно среди разрешённых позиций первой нормальной полосы
            pick = np.linspace(0, len(allowed) - 1, k + 2)[1:-1].round().astype(int) if k < len(allowed) else range(len(allowed))
            idx = [allowed[j] for j in pick]
            for i in idx:
                Image.fromarray(rt[i]).save(os.path.join(dst, 'train', 'good', f'ref_{i:02d}.png'))
            ng = 0
            for p in strips[1:]:
                st = tiles(np.asarray(Image.open(p).convert('RGB')), T)
                for i in allowed[::a.good_every]:
                    Image.fromarray(st[i]).save(os.path.join(dst, 'test', 'good', f'{os.path.basename(p)[:-4]}_t{i:02d}.png'))
                    ng += 1
            nd = 0
            for p in defect.get(fab, []):
                mk = load_mask(p)
                if mk is None:
                    print('нет маски для', os.path.basename(p))
                    continue
                st = tiles(np.asarray(Image.open(p).convert('RGB')), T)
                mt = tiles(mk.astype(np.uint8) * 255, T)
                for i, (t, mm) in enumerate(zip(st, mt)):
                    if i in bad:
                        continue
                    if (mm > 0).sum() >= a.min_defect_px:
                        name = f'{os.path.basename(p)[:-4]}_t{i:02d}'
                        Image.fromarray(t).save(os.path.join(dst, 'test', 'defect', name + '.png'))
                        Image.fromarray(mm).save(os.path.join(dst, 'ground_truth', 'defect', name + '_mask.png'))
                        nd += 1
            print(f'{k}shot/fabric_{fab}: эталонов {len(list(idx))}, нормальных тестовых {ng}, дефектных тестовых {nd} '
                  f'(дефектных полос {len(defect.get(fab, []))})')


if __name__ == '__main__':
    main()
