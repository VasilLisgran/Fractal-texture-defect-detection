# -*- coding: utf-8 -*-
"""
Подготовка AITEX, версия 3 (исправленный дизайн выборки по замечаниям рецензии).

Отличия от v2:
  1. Позиции с фоном (край полосы) исключаются из эталонов и теста по тому же правилу:
     доля пикселей >= 250 или <= 5 на нормальных полосах ткани в среднем больше --bg-frac.
  2. Эталоны: k квадратов из k РАЗНЫХ нормальных полос (не из одной), только с разрешённых позиций.
     --seed 0: первые k полос, середина разрешённых позиций (детерминированно);
     --seed > 0: случайные полосы и случайные разрешённые позиции (для оценки разброса по выбору эталона).
  3. Нормальные тестовые квадраты берутся со ВСЕХ разрешённых позиций равномерно:
     в i-й тестовой полосе берутся позиции allowed[o::good_every], где o = i mod good_every.
     Так у нормы и у дефектов нет систематической разницы в положении вдоль полосы.
  4. --variant ctrl: контрольная выборка, где нормальные квадраты взяты из БЕЗДЕФЕКТНЫХ частей
     дефектных полос (маска пустая, и до ближайшего квадрата с дефектом не меньше --ctrl-gap позиций).
     Это убирает различие «норма и дефект сняты в разных полосах» (с риском неразмеченных дефектов).
Квадрат считается дефектным, если в нём маска (объединение mask, mask1, ...) покрывает >= --min-defect-px пикселей.
Рядом пишется manifest.json со всеми решениями (какие позиции исключены, какие эталоны, сколько квадратов).

Пример:
  python3 prepare_aitex_v3.py --root ~/aitex/archive --out ~/aitex_v3/s0 --shots 1 4 --seed 0
  python3 prepare_aitex_v3.py --root ~/aitex/archive --out ~/aitex_v3/ctrl --shots 1 --seed 0 --variant ctrl
"""
import argparse, glob, json, os, re
from collections import defaultdict
import numpy as np
from PIL import Image

NAME = re.compile(r'^(\d{4})_(\d{3})_(\d{2})\.png$')   # снимок_дефект_ткань


def tiles(arr, T):
    n = arr.shape[1] // T
    return [arr[:T, i * T:(i + 1) * T] for i in range(n)]


def background_fraction(strips, T):
    """Средняя доля фона по позициям; полосы бывают разной длины, поэтому берём общую часть (минимальное число квадратов)."""
    ts = []
    for p in strips:
        g = np.asarray(Image.open(p).convert('L'))
        ts.append(np.array([((x >= 250) | (x <= 5)).mean() for x in tiles(g, T)]))
    n = min(len(t) for t in ts)
    return np.mean([t[:n] for t in ts], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shots', nargs='+', type=int, default=[1, 4])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--variant', choices=['main', 'ctrl'], default='main')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--good-every', type=int, default=4)
    ap.add_argument('--min-defect-px', type=int, default=20)
    ap.add_argument('--bg-frac', type=float, default=0.01)
    ap.add_argument('--ctrl-gap', type=int, default=1, help='ctrl: минимальное расстояние (в квадратах) до квадрата с дефектом')
    a = ap.parse_args()
    root, out, T = os.path.expanduser(a.root), os.path.expanduser(a.out), a.tile

    normal, defect = defaultdict(list), defaultdict(list)
    for p in sorted(glob.glob(os.path.join(root, 'NODefect_images', '*', '*.png'))):
        b = os.path.basename(p); m = NAME.match(b)
        if m and not b.startswith('._'):
            normal[m.group(3)].append(p)
    for p in sorted(glob.glob(os.path.join(root, 'Defect_images', '*.png'))):
        b = os.path.basename(p); m = NAME.match(b)
        if m and not b.startswith('._'):
            defect[m.group(3)].append(p)

    def load_mask(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        files = [f for f in glob.glob(os.path.join(root, 'Mask_images', stem + '_mask*.png'))
                 if not os.path.basename(f).startswith('._')]
        if not files:
            return None
        return np.logical_or.reduce([np.asarray(Image.open(f).convert('L')) > 127 for f in files])

    manifest = {'args': vars(a), 'fabrics': {}}
    for fab in sorted(normal):
        strips = normal[fab]
        fr = background_fraction(strips, T)
        bad = [i for i, f in enumerate(fr) if f > a.bg_frac]
        allowed = [i for i in range(len(fr)) if i not in bad]
        rng = np.random.default_rng(1000 * a.seed + int(fab))
        info = {'bad_positions': {str(i): round(float(fr[i]), 4) for i in bad}, 'allowed': allowed, 'shots': {}}
        # дефектные квадраты (общие для всех k)
        dtiles = []          # (name, tile, mask)
        ctrl_good = []       # (name, tile) - бездефектные квадраты дефектных полос
        n_def_strips = 0
        for p in defect.get(fab, []):
            mk = load_mask(p)
            if mk is None:
                continue
            st = tiles(np.asarray(Image.open(p).convert('RGB')), T)
            mt = tiles(mk.astype(np.uint8) * 255, T)
            px = [int((m > 0).sum()) for m in mt]
            hit = [i for i in allowed if i < len(px) and px[i] >= a.min_defect_px]   # полоса может быть короче
            if hit:
                n_def_strips += 1
            for i in hit:
                dtiles.append((f'{os.path.basename(p)[:-4]}_t{i:02d}', st[i], mt[i]))
            anyd = [i for i in range(len(px)) if px[i] > 0]
            for i in allowed:
                if i < len(px) and px[i] == 0 and all(abs(i - j) > a.ctrl_gap for j in anyd):
                    ctrl_good.append((f'{os.path.basename(p)[:-4]}_t{i:02d}', st[i]))
        for k in a.shots:
            dst = os.path.join(out, f'{k}shot', f'fabric_{fab}')
            for d in ('train/good', 'test/good', 'test/defect', 'ground_truth/defect'):
                os.makedirs(os.path.join(dst, d), exist_ok=True)
            # эталоны: k разных полос
            if a.seed == 0:
                ref_strips = list(range(min(k, len(strips))))
                ref_pos = [allowed[len(allowed) // 2]] * len(ref_strips)
            else:
                ref_strips = sorted(rng.choice(len(strips), size=min(k, len(strips)), replace=False).tolist())
                ref_pos = [int(rng.choice(allowed)) for _ in ref_strips]
            refs = []
            for si, pos in zip(ref_strips, ref_pos):
                tl = tiles(np.asarray(Image.open(strips[si]).convert('RGB')), T)
                t = tl[min(pos, len(tl) - 1)]
                name = f'ref_{os.path.basename(strips[si])[:-4]}_t{pos:02d}.png'
                Image.fromarray(t).save(os.path.join(dst, 'train', 'good', name)); refs.append(name)
            ng = 0
            if a.variant == 'main':
                rest = [s for i, s in enumerate(strips) if i not in ref_strips]
                for i, p in enumerate(rest):
                    st = tiles(np.asarray(Image.open(p).convert('RGB')), T)
                    for pos in allowed[i % a.good_every::a.good_every]:
                        if pos >= len(st):
                            continue
                        Image.fromarray(st[pos]).save(os.path.join(dst, 'test', 'good', f'{os.path.basename(p)[:-4]}_t{pos:02d}.png'))
                        ng += 1
            else:
                for name, t in ctrl_good:
                    Image.fromarray(t).save(os.path.join(dst, 'test', 'good', name + '.png')); ng += 1
            for name, t, m in dtiles:
                Image.fromarray(t).save(os.path.join(dst, 'test', 'defect', name + '.png'))
                Image.fromarray(m).save(os.path.join(dst, 'ground_truth', 'defect', name + '_mask.png'))
            info['shots'][str(k)] = {'references': refs, 'test_good': ng, 'test_defect': len(dtiles),
                                     'defect_strips': n_def_strips}
            print(f'{a.variant} seed {a.seed} {k}shot fabric_{fab}: исключены позиции {bad}; эталоны {refs}; '
                  f'норма {ng}, дефекты {len(dtiles)} из {n_def_strips} полос')
        manifest['fabrics'][fab] = info
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
