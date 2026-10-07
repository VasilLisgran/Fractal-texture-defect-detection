# -*- coding: utf-8 -*-
"""
Сводка пакета прогонов v3: читает все json с оценками снимков из папки результатов и печатает таблицы.
Правило исключения тканей (из протокола): ткань не учитывается, если в ней меньше 5 дефектных тайлов.

Пример:  python3 summarize_v3.py --res v3res
"""
import argparse, glob, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from bootstrap_ci import auroc, fpr_at_tpr

MIN_DEF = 5


def load(path):
    try:
        return json.load(open(path))['records']
    except Exception:
        return None


def per_cat(recs, method, metric=auroc):
    by = {}
    for r in recs:
        if r.get('key') == '__time__' or r.get('method') != method:
            continue
        by.setdefault(r['category'], []).append(r)
    out = {}
    for c, rs in sorted(by.items()):
        lab = [x['label'] for x in rs]; sc = [x['score'] for x in rs]
        out[c] = (metric(sc, lab), sum(lab), len(lab) - sum(lab))
    return out


def times(recs, method):
    return {r['category']: r for r in recs if r.get('key') == '__time__' and r.get('method') == method}


def mean_ok(pc):
    ok = [c for c, v in pc.items() if v[1] >= MIN_DEF]
    return (np.mean([pc[c][0] for c in ok]) if ok else np.nan), ok


def row(name, pc, cats):
    m, ok = mean_ok(pc)
    cells = ' '.join(f'{pc[c][0]:.3f}' if c in pc else '  -  ' for c in cats)
    return f'{name:22s} {cells}   | {m:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='v3res')
    a = ap.parse_args()
    R = lambda f: load(os.path.join(a.res, f))

    def spec(tag):
        return [(f'ours_{tag}.json', 'plain', 'plain'), (f'light_{tag}.json', 'affine', 'light (fit only)'),
                (f'ours_{tag}.json', 'affine_iso', 'OURS full'), (f'pcA_{tag}.json', 'patchcore_A', 'PatchCore A'),
                (f'pcB_{tag}.json', 'patchcore_B', 'PatchCore B'), (f'padim_{tag}.json', 'padim', 'PaDiM')]

    for tag, title in (('s0_1', 'ОСНОВНОЙ, 1 эталон'), ('s0_4', 'ОСНОВНОЙ, 4 эталона'), ('ctrl_1', 'КОНТРОЛЬ (норма из дефектных полос), 1 эталон')):
        print(f'\n=== {title}: AUROC по изображению ===')
        tabs = {}
        for f, m, name in spec(tag):
            recs = R(f)
            if recs:
                tabs[name] = per_cat(recs, m)
        if not tabs:
            print('  нет данных'); continue
        cats = sorted(set().union(*[set(t) for t in tabs.values()]))
        any_t = next(iter(tabs.values()))
        print('состав: ' + ', '.join(f'{c[-2:]}: деф {any_t[c][1]}, норм {any_t[c][2]}' for c in cats if c in any_t))
        print(f'{"метод":22s} ' + ' '.join(f'{c[-2:]:>5s}' for c in cats) + '   | среднее (ткани с >=5 деф.)')
        for name, pc in tabs.items():
            print(row(name, pc, cats))
        # пиксели и рабочая точка
        print('  AUROC по пикселям (среднее по учитываемым тканям) и FPR при полноте 95 % (среднее):')
        for f, m, name in spec(tag):
            recs = R(f)
            if not recs:
                continue
            tm = times(recs, m); pc = per_cat(recs, m); _, ok = mean_ok(pc)
            pix = [tm[c].get('pixel_AUROC') for c in ok if c in tm and tm[c].get('pixel_AUROC') is not None]
            fp = per_cat(recs, m, fpr_at_tpr)
            print(f'    {name:22s} пиксели {np.mean(pix) if pix else float("nan"):.3f}   FPR95 {np.mean([fp[c][0] for c in ok]):.3f}')
        # правило решения
        if 'OURS full' in tabs and ('PatchCore A' in tabs or 'PatchCore B' in tabs):
            ours = tabs['OURS full']; mo, ok = mean_ok(ours)
            best = max([n for n in ('PatchCore A', 'PatchCore B') if n in tabs], key=lambda n: mean_ok(tabs[n])[0])
            pb, _ = mean_ok(tabs[best])
            wins = sum(ours[c][0] > tabs[best][c][0] for c in ok if c in tabs[best])
            ties = sum(ours[c][0] == tabs[best][c][0] for c in ok if c in tabs[best])
            print(f'  ПРАВИЛО: лучший PatchCore = {best} ({pb:.3f}); наш {mo:.3f}; не хуже в среднем: {mo >= pb}; '
                  f'побед {wins} из {len(ok)} (ничьих {ties}); половина: {wins >= len(ok) / 2}')

    print('\n=== РАЗБРОС ПО ВЫБОРУ ЭТАЛОНА (1 эталон, случайные полоса и позиция) ===')
    rows = []
    for s in range(0, 6):
        tag = f's{s}_1'
        vals = {}
        for f, m, name in spec(tag)[:4]:
            recs = R(f)
            if recs:
                vals[name] = mean_ok(per_cat(recs, m))[0]
        if vals:
            rows.append((s, vals))
            print(f'  seed {s}: ' + '  '.join(f'{k} {v:.3f}' for k, v in vals.items()) +
                  (f'   | наш - PatchCore A = {vals["OURS full"] - vals["PatchCore A"]:+.3f}' if 'OURS full' in vals and 'PatchCore A' in vals else ''))
    for name in ('plain', 'light (fit only)', 'OURS full', 'PatchCore A'):
        v = [r[1][name] for r in rows if name in r[1]]
        if len(v) > 1:
            print(f'  {name:22s} среднее {np.mean(v):.3f}, ст. откл. {np.std(v, ddof=1):.3f}, мин {min(v):.3f}, макс {max(v):.3f}  (n={len(v)})')
    d = [r[1]['OURS full'] - r[1]['PatchCore A'] for r in rows if 'OURS full' in r[1] and 'PatchCore A' in r[1]]
    if len(d) > 1:
        print(f'  разница наш - PatchCore A по выборам эталона: среднее {np.mean(d):+.3f}, мин {min(d):+.3f}, макс {max(d):+.3f}, '
              f'положительна в {sum(x > 0 for x in d)} из {len(d)}')

    print('\n=== АБЛЯЦИЯ И ЧУВСТВИТЕЛЬНОСТЬ (основной набор, 1 эталон; среднее AUROC по учитываемым тканям) ===')
    for f in sorted(glob.glob(os.path.join(a.res, 'abl_*.json')) + glob.glob(os.path.join(a.res, 'sens_*.json'))):
        recs = load(f)
        if not recs:
            continue
        for m in sorted({r['method'] for r in recs}):
            pc = per_cat(recs, m); mo, ok = mean_ok(pc)
            tm = times(recs, m)
            t3 = tm.get('fabric_03', {}).get('seconds_total')
            print(f'  {os.path.basename(f)[:-5]:28s} {m:11s} AUROC {mo:.3f}   ' + ' '.join(f'{c[-2:]}:{pc[c][0]:.3f}' for c in ok) +
                  (f'   время ткани 03: {t3:.0f} с' if t3 else ''))

    print('\n=== ОБЩЕЕ УСИЛЕНИЕ ЯРКОСТИ ТЕСТОВЫХ СНИМКОВ (проверка инвариантности; без обрезки значений) ===')
    for f in sorted(glob.glob(os.path.join(a.res, 'gain_*.json'))):
        recs = load(f)
        if not recs:
            continue
        for m in sorted({r['method'] for r in recs}):
            mo, _ = mean_ok(per_cat(recs, m))
            print(f'  {os.path.basename(f)[:-5]:20s} {m:11s} AUROC {mo:.3f}')

    print('\n=== ВРЕМЯ (ткань 03, основной набор, 1 эталон) ===')
    for f, m, name in spec('s0_1'):
        recs = R(f)
        if recs:
            t = times(recs, m).get('fabric_03')
            if t:
                n = t.get('n_images')
                print(f'  {name:22s} {t.get("seconds_fit_test", t.get("seconds_total")):.0f} с' + (f' на {n} снимков' if n else ''))


if __name__ == '__main__':
    main()
