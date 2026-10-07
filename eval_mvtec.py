# -*- coding: utf-8 -*-
"""
Оценка обнаружения дефектов текстур по эталонным образцам (идея: сопоставление блоков в духе PIFS).

Три варианта сопоставления блока проверяемого изображения с блоками эталонов:
  plain       - простое сравнение (L2 по пикселям), без подгонки яркости и без поворотов;
  affine      - с подгонкой контраста s и сдвига яркости o, без поворотов;
  affine_iso  - с подгонкой s, o и перебором 8 изометрий (поворот/отражение).
Оценка блока = остаточная MSE после лучшего совпадения; карта оценок = карта дефектов.

Запуск (пример):
  python eval_mvtec.py --root /путь/к/mvtec_anomaly_detection --cat carpet grid leather tile wood \
                       --shots 1 5 --size 256 --block 16 --stride 4

Структура папок MVTec AD:  <root>/<cat>/train/good/*.png, <root>/<cat>/test/<тип>/*.png,
                           <root>/<cat>/ground_truth/<тип>/*_mask.png
Лицензия набора MVTec AD: CC BY-NC-SA 4.0 (некоммерческое использование).
"""
import argparse, glob, os, time
import numpy as np
from PIL import Image


def isometries(b):
    out = []
    for k in range(4):
        out.append(np.rot90(b, k))
    f = np.fliplr(b)
    for k in range(4):
        out.append(np.rot90(f, k))
    return out


def blocks(g, B, step):
    H, W = g.shape
    P, pos = [], []
    for y in range(0, H - B + 1, step):
        for x in range(0, W - B + 1, step):
            P.append(g[y:y + B, x:x + B].ravel())
            pos.append((y, x))
    return np.array(P, np.float32), np.array(pos)


def build_bank(refs, B, step, method):
    cand = []
    for g in refs:
        for y in range(0, g.shape[0] - B + 1, step):
            for x in range(0, g.shape[1] - B + 1, step):
                b = g[y:y + B, x:x + B]
                if method == 'affine_iso':
                    cand.extend(i.ravel() for i in isometries(b))
                else:
                    cand.append(b.ravel())
    C = np.array(cand, np.float32)
    if method == 'plain':
        return C, (C ** 2).sum(1)
    C = C - C.mean(1, keepdims=True)
    return C, (C ** 2).sum(1)


def local_median(grid, win):
    """Медиана по окну win x win блоков (фон яркости вокруг каждого блока)."""
    from numpy.lib.stride_tricks import sliding_window_view
    p = win // 2
    gp = np.pad(grid, p, mode='edge')
    w = sliding_window_view(gp, (win, win))
    return np.median(w.reshape(grid.shape[0], grid.shape[1], -1), axis=2)


def score_map(g, bank, method, B, step, smin, smax, norm, mean_w=0.0, mean_win=15):
    C, V = bank
    R, rp = blocks(g, B, step)
    n2 = float(B * B)
    sc = np.zeros(len(R), np.float32)
    CH = 256
    for s in range(0, len(R), CH):
        r = R[s:s + CH]
        if method == 'plain':
            d2 = (r ** 2).sum(1)[:, None] + V[None, :] - 2 * r @ C.T
            e = d2.min(1) / n2
        else:
            rc = r - r.mean(1, keepdims=True)
            vr = (rc ** 2).sum(1)
            cov = rc @ C.T
            with np.errstate(divide='ignore', invalid='ignore'):
                sv = np.where(V[None, :] > 1e-6, cov / (V[None, :] + 1e-9), 0.0)
            if smax > 0:
                sv = np.clip(sv, smin, smax)          # как в Core.py: контраст в пределах [smin, smax]
            res = (vr[:, None] - 2 * sv * cov + sv ** 2 * V[None, :]) / n2
            if norm:
                res = res / np.maximum(sv, 1e-3) ** 2   # ошибка в единицах эталона
            e = res.min(1)
        sc[s:s + CH] = e
    ny = len(range(0, g.shape[0] - B + 1, step))
    nx = len(range(0, g.shape[1] - B + 1, step))
    grid = sc.reshape(ny, nx)
    if method != 'plain' and mean_w > 0:
        # локальная яркость: средняя яркость блока против медианы по окрестности.
        # Плавный перепад освещения почти не меняет разницу, а локальное пятно меняет сильно.
        mr = R.mean(1).reshape(ny, nx)
        bg = local_median(mr, mean_win)
        delta = (mr - bg) / np.maximum(bg, 0.05)      # относительная разница, не зависит от общего усиления яркости
        grid = grid + mean_w * delta ** 2
    m = np.full(g.shape, np.nan, np.float32)
    h = B // 2
    for i in range(ny):
        for j in range(nx):
            y, x = rp[i * nx + j]
            m[y + h - step // 2:y + h + step // 2, x + h - step // 2:x + h + step // 2] = grid[i, j]
    fill = np.nanmin(m)
    return np.where(np.isnan(m), fill, m)


def auroc(score, label):
    s = np.asarray(score, np.float64).ravel()
    y = np.asarray(label, bool).ravel()
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # поправка на совпадающие значения
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def load_gray(path, size):
    return np.asarray(Image.open(path).convert('L').resize((size, size), Image.BILINEAR), np.float32) / 255.0


def apply_shift(g, m, shift):
    n = g.shape[0]
    if shift == 'light':
        yy, xx = np.indices(g.shape)
        g = g * (0.6 + 0.8 * xx / n) + 0.15 * yy / n     # плавный перепад освещения
    elif shift == 'rot90':
        g = np.ascontiguousarray(np.rot90(g)); m = np.ascontiguousarray(np.rot90(m))
    elif shift.startswith('gain'):
        g = g * float(shift[4:])                          # общее усиление яркости, например gain0.7 (без обрезки)
    return g.astype(np.float32), m


SCORE_LOG = []   # оценки каждого снимка (для доверительных интервалов)


def run_category(root, cat, shots, size, B, step, smin, smax, norm, shift, methods, viz_dir, mean_w=0.0, mean_win=15):
    train = sorted(glob.glob(os.path.join(root, cat, 'train', 'good', '*.*')))
    tests = []
    for d in sorted(os.listdir(os.path.join(root, cat, 'test'))):
        for p in sorted(glob.glob(os.path.join(root, cat, 'test', d, '*.*'))):
            tests.append((d, p))
    print(f'\n=== {cat}: эталонов в train/good: {len(train)}, тестовых изображений: {len(tests)}, сдвиг условий: {shift} ===')
    imgs = []
    for d, p in tests:
        g = load_gray(p, size)
        if d == 'good':
            m = np.zeros((size, size), bool)
        else:
            mp = os.path.join(root, cat, 'ground_truth', d, os.path.splitext(os.path.basename(p))[0] + '_mask.png')
            m = np.asarray(Image.open(mp).convert('L').resize((size, size), Image.NEAREST)) > 127
        g, m = apply_shift(g, m, shift)
        imgs.append((d, g, m, f'{d}/{os.path.basename(p)}'))
    results = {}
    for k in shots:
        refs = [load_gray(p, size) for p in train[:k]]
        for meth in methods:
            t0 = time.time()
            bank = build_bank(refs, B, step, meth)
            img_scores, img_top, img_labels, pix_s, pix_l, types = [], [], [], [], [], []
            keys = []
            for idx, (d, g, m, key) in enumerate(imgs):
                keys.append(key)
                sm = score_map(g, bank, meth, B, step, smin, smax, norm, mean_w, mean_win)
                img_scores.append(sm.max())
                flat = np.sort(sm.ravel())
                img_top.append(flat[-max(1, len(flat) // 100):].mean())   # среднее по 1% самых плохих пикселей
                img_labels.append(d != 'good')
                types.append(d)
                pix_s.append(sm[::2, ::2].ravel())
                pix_l.append(m[::2, ::2].ravel())
                if viz_dir and d != 'good' and idx % 20 == 0:
                    os.makedirs(viz_dir, exist_ok=True)
                    v = (255 * (sm - sm.min()) / (np.ptp(sm) + 1e-9)).astype(np.uint8)
                    row = np.concatenate([(255 * g).astype(np.uint8), v, (255 * m).astype(np.uint8)], 1)
                    Image.fromarray(row).save(os.path.join(viz_dir, f'{cat}_{meth}_{k}shot_{idx:03d}.png'))
            ia = auroc(img_scores, img_labels)
            it = auroc(img_top, img_labels)
            pa = auroc(np.concatenate(pix_s), np.concatenate(pix_l))
            results[(k, meth)] = (ia, pa, it)
            dt = time.time() - t0
            for key, sc, lab, ty in zip(keys, img_scores, img_labels, types):
                SCORE_LOG.append({'category': cat, 'method': meth, 'shots': int(k), 'key': key,
                                  'label': int(lab), 'type': ty, 'score': float(sc)})
            SCORE_LOG.append({'category': cat, 'method': meth, 'shots': int(k), 'key': '__time__',
                              'seconds_total': float(dt), 'n_images': len(keys),
                              'pixel_AUROC': float(pa), 'image_AUROC': float(ia)})
            print(f'  эталонов {k}  {meth:11s} AUROC(изобр., max) {ia:.3f}  (изобр., топ-1%) {it:.3f}   AUROC(пиксель) {pa:.3f}   ({time.time() - t0:.0f} с)', flush=True)
            sc_arr, ty_arr = np.array(img_scores), np.array(types)
            per = []
            for d in sorted(set(types) - {'good'}):
                sel = (ty_arr == 'good') | (ty_arr == d)
                per.append(f'{d} {auroc(sc_arr[sel], ty_arr[sel] != "good"):.2f}')
            print('      по типам дефектов (AUROC изобр.): ' + ', '.join(per), flush=True)
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--cat', nargs='+', default=['carpet', 'grid', 'leather', 'tile', 'wood'])
    ap.add_argument('--shots', nargs='+', type=int, default=[1, 5])
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--block', type=int, default=16)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--smin', type=float, default=0.6, help='нижняя граница контраста s')
    ap.add_argument('--smax', type=float, default=1.7, help='верхняя граница контраста s (0 - без ограничения)')
    ap.add_argument('--norm', action='store_true', help='нормировать ошибку на s^2')
    ap.add_argument('--shift', default='none', help='сдвиг условий проверяемых снимков: none, light, rot90 или gainA (например gain0.7)')
    ap.add_argument('--methods', nargs='+', default=['plain', 'affine', 'affine_iso'])
    ap.add_argument('--mean-w', type=float, default=0.0, help='вес поправки на локальную яркость (0 - выключено)')
    ap.add_argument('--mean-win', type=int, default=15, help='окно (в блоках) для фона яркости')
    ap.add_argument('--viz', default='', help='папка для сохранения тепловых карт')
    ap.add_argument('--save-scores', default='', help='json-файл для оценок каждого снимка (нужен для доверительных интервалов)')
    a = ap.parse_args()
    allres = {}
    for c in a.cat:
        allres[c] = run_category(a.root, c, a.shots, a.size, a.block, a.stride, a.smin, a.smax, a.norm, a.shift, a.methods, a.viz or None, a.mean_w, a.mean_win)
    if a.save_scores:
        import json
        meta = {k: v for k, v in vars(a).items()}
        with open(a.save_scores, 'w') as f:
            json.dump({'source': 'ours', 'meta': meta, 'records': SCORE_LOG}, f, ensure_ascii=False)
        print('оценки снимков сохранены в', a.save_scores)
    print('\n=== Сводка: среднее по категориям ===')
    for k in a.shots:
        for m in a.methods:
            ia = np.nanmean([allres[c][(k, m)][0] for c in a.cat])
            pa = np.nanmean([allres[c][(k, m)][1] for c in a.cat])
            it = np.nanmean([allres[c][(k, m)][2] for c in a.cat])
            print(f'  эталонов {k}  {m:11s} AUROC(изобр., max) {ia:.3f}  (изобр., топ-1%) {it:.3f}  AUROC(пикс.) {pa:.3f}')