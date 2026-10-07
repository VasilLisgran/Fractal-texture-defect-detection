# -*- coding: utf-8 -*-
"""
Убирает из УЖЕ сохранённых файлов оценок тестовые квадраты на краю полосы, где в кадр попал фон.
Модели повторно не запускаются: эталоны краёв не содержали, меняется только состав теста.

Позиция считается краевой, если на нормальных полосах ткани доля пикселей фона (>= 250 или <= 5)
в ней в среднем больше 1 % (правило из PROTOCOL.md, поправка 1).

Пример:
  python3 filter_edges.py --root ~/aitex/archive --files ours_aitex_1.json ours_aitex_4.json abl_aitex.json \
      pc_A_1_scores.json pc_B_1_scores.json pc_A_4_scores.json pc_B_4_scores.json
Создаёт рядом файлы с приставкой noedge_ и печатает AUROC по тканям до и после.
"""
import argparse, glob, json, os, re
import numpy as np
from PIL import Image
from bootstrap_ci import auroc

NAME = re.compile(r'^(\d{4})_(\d{3})_(\d{2})\.png$')
TILE = re.compile(r'_t(\d+)\.png$')


def edge_positions(root, T=256, frac=0.01):
    by_fab = {}
    for p in glob.glob(os.path.join(root, 'NODefect_images', '*', '*.png')):
        m = NAME.match(os.path.basename(p))
        if m and not os.path.basename(p).startswith('._'):
            by_fab.setdefault(m.group(3), []).append(p)
    out = {}
    for fab, strips in sorted(by_fab.items()):
        acc = None
        for p in strips:
            g = np.asarray(Image.open(p).convert('L'))
            n = g.shape[1] // T
            f = np.array([((g[:T, i * T:(i + 1) * T] >= 250) | (g[:T, i * T:(i + 1) * T] <= 5)).mean() for i in range(n)])
            acc = f if acc is None else acc + f
        acc /= len(strips)
        out[f'fabric_{fab}'] = [i for i, v in enumerate(acc) if v > frac]
        if out[f'fabric_{fab}']:
            print(f'fabric_{fab}: краевые позиции {out[f"fabric_{fab}"]} (доля фона ' +
                  ', '.join(f'{acc[i]:.2f}' for i in out[f'fabric_{fab}']) + ')')
    return out


def summary(records, title):
    groups = {}
    for r in records:
        if r.get('key') == '__time__':
            continue
        groups.setdefault((r['method'], r.get('shots'), r['category']), []).append(r)
    by_m = {}
    for (m, k, c), rs in sorted(groups.items()):
        by_m.setdefault((m, k), {})[c] = (auroc([x['score'] for x in rs], [x['label'] for x in rs]),
                                          sum(x['label'] for x in rs), len(rs))
    for (m, k), d in by_m.items():
        cats = [c for c in sorted(d) if c != 'fabric_05']
        mean = np.mean([d[c][0] for c in cats])
        print(f'  {title:6s} {m:12s} эталонов {k}:  среднее без 05 = {mean:.3f}   ' +
              '  '.join(f'{c[-2:]}:{d[c][0]:.3f}' for c in sorted(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='папка archive набора AITEX')
    ap.add_argument('--files', nargs='+', required=True)
    a = ap.parse_args()
    bad = edge_positions(os.path.expanduser(a.root))
    for fn in a.files:
        d = json.load(open(fn))
        recs = d['records']
        keep, removed = [], 0
        for r in recs:
            if r.get('key') != '__time__':
                m = TILE.search(r['key'])
                if m and int(m.group(1)) in bad.get(r['category'], []):
                    removed += 1
                    continue
            keep.append(r)
        print(f'\n{fn}: убрано краевых снимков {removed}')
        summary(recs, 'было')
        summary(keep, 'стало')
        d['records'] = keep
        d['edge_filter'] = {k: v for k, v in bad.items()}
        out = os.path.join(os.path.dirname(fn), 'noedge_' + os.path.basename(fn))
        json.dump(d, open(out, 'w'), ensure_ascii=False)
        print('  сохранено в', out)


if __name__ == '__main__':
    main()
