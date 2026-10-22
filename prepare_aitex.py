# -*- coding: utf-8 -*-
"""
Сводка пакета прогонов v3: читает все json с оценками снимков из папки результатов и печатает таблицы.
Правило исключения тканей (из протокола): ткань не учитывается, если в ней меньше 5 дефектных тайлов.

Скрипт ничего не запускает и не обучает: он только читает готовые оценки каждого тайла (их записывают
eval_mvtec.py и run_patchcore.py) и считает по ним метрики. Доверительные интервалы здесь не строятся,
их считает bootstrap_ci.py.

Пример:  python3 summarize_v3.py --res v3res
"""
import argparse, glob, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
from bootstrap_ci import auroc, fpr_at_tpr

# Ткань попадает в средние, только если в ней не меньше стольких дефектных тайлов
# (по этому правилу из таблиц выпадает ткань 05: в ней один дефектный тайл).
MIN_DEF = 5


def load(path):
    """Читает json с оценками тайлов и возвращает список записей; None, если файла нет или он повреждён."""
    try:
        return json.load(open(path))['records']
    except Exception:
        return None


def per_cat(recs, method, metric=auroc):
    """Считает метрику отдельно по каждой ткани для одного метода.

    recs   - записи из json (служебные записи __time__ пропускаются);
    method - какой метод брать, если в файле их несколько (например, plain и affine_iso лежат в одном файле);
    metric - функция метрики: AUROC по умолчанию, можно передать fpr_at_tpr.

    Возвращает {ткань: (значение метрики, число дефектных тайлов, число нормальных тайлов)}.
    """
    by = {}
    for r in recs:
        # пропускаем служебные записи со временем и записи других методов
        if r.get('key') == '__time__' or r.get('method') != method:
            continue
        by.setdefault(r['category'], []).append(r)   # группируем записи по ткани
    out = {}
    for c, rs in sorted(by.items()):
        lab = [x['label'] for x in rs]; sc = [x['score'] for x in rs]   # метки (0 норма, 1 дефект) и оценки тайлов
        out[c] = (metric(sc, lab), sum(lab), len(lab) - sum(lab))       # метрика, дефектных, нормальных
    return out


def times(recs, method):
    """Служебные записи __time__ выбранного метода: {ткань: запись со временем и контрольными AUROC}."""
    return {r['category']: r for r in recs if r.get('key') == '__time__' and r.get('method') == method}


def mean_ok(pc):
    """Среднее метрики по тканям, где дефектных тайлов не меньше MIN_DEF.

    pc - результат per_cat. Возвращает (среднее, список учтённых тканей); nan, если таких тканей нет.
    """
    ok = [c for c, v in pc.items() if v[1] >= MIN_DEF]
    return (np.mean([pc[c][0] for c in ok]) if ok else np.nan), ok


def row(name, pc, cats):
    """Строка таблицы: имя метода, значение по каждой ткани из cats (прочерк, если данных нет) и среднее после '|'.

    В строке показаны все ткани, а среднее считается только по учитываемым (см. mean_ok).
    """
    m, ok = mean_ok(pc)
    cells = ' '.join(f'{pc[c][0]:.3f}' if c in pc else '  -  ' for c in cats)
    return f'{name:22s} {cells}   | {m:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='v3res')     # папка с json-файлами результатов
    a = ap.parse_args()
    R = lambda f: load(os.path.join(a.res, f))    # прочитать файл из папки результатов по имени

    def spec(tag):
        # Какие файлы и методы нужны для набора с меткой tag. Каждая тройка: (файл, метод внутри файла, название в таблице).
        # plain и OURS full лежат в одном файле ours_*.json и различаются полем method.
        return [(f'ours_{tag}.json', 'plain', 'plain'), (f'light_{tag}.json', 'affine', 'light (fit only)'),
                (f'ours_{tag}.json', 'affine_iso', 'OURS full'), (f'pcA_{tag}.json', 'patchcore_A', 'PatchCore A'),
                (f'pcB_{tag}.json', 'patchcore_B', 'PatchCore B'), (f'padim_{tag}.json', 'padim', 'PaDiM')]

    #Три основные таблицы: основной дизайн (1 и 4 эталона) и контроль (нормальные тайлы взяты из дефектных полос) ----
    for tag, title in (('s0_1', 'ОСНОВНОЙ, 1 эталон'), ('s0_4', 'ОСНОВНОЙ, 4 эталона'), ('ctrl_1', 'КОНТРОЛЬ (норма из дефектных полос), 1 эталон')):
        print(f'\n=== {title}: AUROC по изображению ===')
        # tabs: {название метода: результат per_cat}; методы, для которых нет файла, пропускаются
        tabs = {}
        for f, m, name in spec(tag):
            recs = R(f)
            if recs:
                tabs[name] = per_cat(recs, m)
        if not tabs:
            print('  нет данных'); continue
        cats = sorted(set().union(*[set(t) for t in tabs.values()]))   # все ткани, встретившиеся хотя бы у одного метода
        any_t = next(iter(tabs.values()))                              # состав набора (число тайлов) один для всех методов
        print('состав: ' + ', '.join(f'{c[-2:]}: деф {any_t[c][1]}, норм {any_t[c][2]}' for c in cats if c in any_t))
        print(f'{"метод":22s} ' + ' '.join(f'{c[-2:]:>5s}' for c in cats) + '   | среднее (ткани с >=5 деф.)')
        for name, pc in tabs.items():
            print(row(name, pc, cats))
        # AUROC по пикселям (качество локализации) и FPR95 (доля ложных тревог при обнаружении 95 % дефектов)
        print('  AUROC по пикселям (среднее по учитываемым тканям) и FPR при полноте 95 % (среднее):')
        for f, m, name in spec(tag):
            recs = R(f)
            if not recs:
                continue
            tm = times(recs, m); pc = per_cat(recs, m); _, ok = mean_ok(pc)
            # пиксельный AUROC не считается по оценкам тайлов: его записала модель в служебную запись __time__
            pix = [tm[c].get('pixel_AUROC') for c in ok if c in tm and tm[c].get('pixel_AUROC') is not None]
            fp = per_cat(recs, m, fpr_at_tpr)   # та же функция, но с другой метрикой: FPR95 вместо AUROC
            print(f'    {name:22s} пиксели {np.mean(pix) if pix else float("nan"):.3f}   FPR95 {np.mean([fp[c][0] for c in ok]):.3f}')
        # Правило решения из протокола: наш метод не хуже лучшего PatchCore в среднем И выигрывает не меньше чем на половине тканей
        if 'OURS full' in tabs and ('PatchCore A' in tabs or 'PatchCore B' in tabs):
            ours = tabs['OURS full']; mo, ok = mean_ok(ours)
            # лучший из двух PatchCore по среднему AUROC (чтобы конкурента нельзя было счесть ослабленным)
            best = max([n for n in ('PatchCore A', 'PatchCore B') if n in tabs], key=lambda n: mean_ok(tabs[n])[0])
            pb, _ = mean_ok(tabs[best])
            wins = sum(ours[c][0] > tabs[best][c][0] for c in ok if c in tabs[best])    # ткани, где мы строго выше
            ties = sum(ours[c][0] == tabs[best][c][0] for c in ok if c in tabs[best])   # ничьи; в победы не входят
            print(f'  ПРАВИЛО: лучший PatchCore = {best} ({pb:.3f}); наш {mo:.3f}; не хуже в среднем: {mo >= pb}; '
                  f'побед {wins} из {len(ok)} (ничьих {ties}); половина: {wins >= len(ok) / 2}')

    # Разброс по выбору эталона: прогоны s0..s5 с другой полосой и позицией эталона (один эталон) ----
    # Если файлов s1_1..s5_1 в папке нет, блок ничего не печатает.
    print('\n=== РАЗБРОС ПО ВЫБОРУ ЭТАЛОНА (1 эталон, случайные полоса и позиция) ===')
    rows = []
    for s in range(0, 6):
        tag = f's{s}_1'
        vals = {}
        for f, m, name in spec(tag)[:4]: # первые четыре метода: plain, light, OURS full, PatchCore A
            recs = R(f)
            if recs:
                vals[name] = mean_ok(per_cat(recs, m))[0]   # среднее AUROC по учитываемым тканям
        if vals:
            rows.append((s, vals))
            print(f'  seed {s}: ' + '  '.join(f'{k} {v:.3f}' for k, v in vals.items()) +
                  (f'   | наш - PatchCore A = {vals["OURS full"] - vals["PatchCore A"]:+.3f}' if 'OURS full' in vals and 'PatchCore A' in vals else ''))
    # сводка по выбору эталона: среднее, стандартное отклонение, минимум и максимум для каждого метода
    for name in ('plain', 'light (fit only)', 'OURS full', 'PatchCore A'):
        v = [r[1][name] for r in rows if name in r[1]]
        if len(v) > 1:
            print(f'  {name:22s} среднее {np.mean(v):.3f}, ст. откл. {np.std(v, ddof=1):.3f}, мин {min(v):.3f}, макс {max(v):.3f}  (n={len(v)})')
    # разность «наш - PatchCore A» при разных выборах эталона и в скольких выборах она положительна
    d = [r[1]['OURS full'] - r[1]['PatchCore A'] for r in rows if 'OURS full' in r[1] and 'PatchCore A' in r[1]]
    if len(d) > 1:
        print(f'  разница наш - PatchCore A по выборам эталона: среднее {np.mean(d):+.3f}, мин {min(d):+.3f}, макс {max(d):+.3f}, '
              f'положительна в {sum(x > 0 for x in d)} из {len(d)}')

    # Абляция и чувствительность: какой вклад вносит каждая деталь метода (файлы abl_*.json и sens_*.json) ----
    print('\n=== АБЛЯЦИЯ И ЧУВСТВИТЕЛЬНОСТЬ (основной набор, 1 эталон; среднее AUROC по учитываемым тканям) ===')
    for f in sorted(glob.glob(os.path.join(a.res, 'abl_*.json')) + glob.glob(os.path.join(a.res, 'sens_*.json'))):
        recs = load(f)
        if not recs:
            continue
        for m in sorted({r['method'] for r in recs}):   # в файле может быть несколько методов
            pc = per_cat(recs, m); mo, ok = mean_ok(pc)
            tm = times(recs, m)
            t3 = tm.get('fabric_03', {}).get('seconds_total')   # время на ткани 03, если записано
            print(f'  {os.path.basename(f)[:-5]:28s} {m:11s} AUROC {mo:.3f}   ' + ' '.join(f'{c[-2:]}:{pc[c][0]:.3f}' for c in ok) +
                  (f'   время ткани 03: {t3:.0f} с' if t3 else ''))

    # Проверка инвариантности: тестовые снимки умножены на общий множитель яркости (файлы gain_*.json) ----
    # Если метод не зависит от общего усиления яркости, AUROC почти не должен меняться.
    print('\n=== ОБЩЕЕ УСИЛЕНИЕ ЯРКОСТИ ТЕСТОВЫХ СНИМКОВ (проверка инвариантности; без обрезки значений) ===')
    for f in sorted(glob.glob(os.path.join(a.res, 'gain_*.json'))):
        recs = load(f)
        if not recs:
            continue
        for m in sorted({r['method'] for r in recs}):
            mo, _ = mean_ok(per_cat(recs, m))
            print(f'  {os.path.basename(f)[:-5]:20s} {m:11s} AUROC {mo:.3f}')

    #Время работы на ткани 03 (основной набор, 1 эталон): для PatchCore включает построение банка по эталонам ----
    print('\n=== ВРЕМЯ (ткань 03, основной набор, 1 эталон) ===')
    for f, m, name in spec('s0_1'):
        recs = R(f)
        if recs:
            t = times(recs, m).get('fabric_03')
            if t:
                n = t.get('n_images') # число тестовых снимков, если записано
                print(f'  {name:22s} {t.get("seconds_fit_test", t.get("seconds_total")):.0f} с' + (f' на {n} снимков' if n else ''))


if __name__ == '__main__':
    main()
