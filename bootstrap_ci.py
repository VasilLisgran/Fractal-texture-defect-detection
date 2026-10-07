# -*- coding: utf-8 -*-
"""
Парный бутстреп для сравнения двух методов по AUROC на уровне изображения.

Берёт файлы оценок снимков (eval_mvtec.py --save-scores и run_patchcore.py --save-scores),
сопоставляет снимки по ключу <тип>/<имя файла> и много раз перевыбирает тестовые снимки
с возвращением (отдельно нормальные и дефектные, чтобы их число не менялось).
На каждой перевыборке AUROC обоих методов считается на ОДНИХ И ТЕХ ЖЕ снимках.

Выводит для каждой категории и для среднего по категориям:
  AUROC метода A и B с 95% интервалами, разницу A - B с 95% интервалом
  и долю перевыборок, где A не лучше B (односторонний бутстреп-p).

Режим --by-strip: квадраты одной полосы ткани похожи друг на друга, поэтому перевыбираются не отдельные
квадраты, а целые ПОЛОСЫ (кластерный бутстреп; полоса определяется по имени файла до суффикса _tNN).
Интервалы при этом шире и честнее. Без этого флага квадраты считаются независимыми (интервалы оптимистичны).

Пример (наш метод против PatchCore-A при одном эталоне, без ткани 05):
  python3 bootstrap_ci.py --a ours_aitex_1.json --a-method affine_iso --a-shots 1 \
                          --b pc_aitex_A_1.json --b-method patchcore_A --exclude fabric_05
"""
import argparse, json, re
import numpy as np

STRIP = re.compile(r'^(?:[^/]+)/(.+)_t\d+\.png$')


def strip_id(key):
    m = STRIP.match(key)
    return m.group(1) if m else key


def auroc(score, label):
    s = np.asarray(score, np.float64)
    y = np.asarray(label, bool)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    sv = s[order]; i = 0
    while i < len(sv):                       # средние ранги для совпадающих значений
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return (ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fpr_at_tpr(score, label, tpr=0.95):
    """Доля ложных тревог (норма с оценкой >= порога) при пороге, который ловит не меньше tpr дефектов."""
    s = np.asarray(score, np.float64); y = np.asarray(label, bool)
    pos = np.sort(s[y]); neg = s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    k = int(np.floor((1 - tpr) * len(pos)))
    thr = pos[k]
    return float((neg >= thr).mean())


METRIC = auroc
LOWER_BETTER = False
MNAME = 'AUROC'


def load(path, method, shots):
    d = json.load(open(path))
    out = {}
    for r in d['records']:
        if r.get('key') == '__time__' or r.get('method') != method:
            continue
        if shots is not None and r.get('shots') is not None and int(r['shots']) != shots:
            continue
        out.setdefault(r['category'], {})[r['key']] = (int(r['label']), float(r['score']))
    return out


def ci(x):
    x = np.asarray(x); x = x[~np.isnan(x)]
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True); ap.add_argument('--a-method', required=True); ap.add_argument('--a-shots', type=int)
    ap.add_argument('--b', required=True); ap.add_argument('--b-method', required=True); ap.add_argument('--b-shots', type=int)
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--by-strip', action='store_true', help='перевыбирать целые полосы, а не отдельные квадраты')
    ap.add_argument('--metric', choices=['auroc', 'fpr95'], default='auroc',
                    help='auroc или fpr95 (доля ложных тревог при полноте 95 %%; меньше - лучше)')
    ap.add_argument('--n', type=int, default=2000, help='число перевыборок')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    global METRIC, LOWER_BETTER, MNAME
    if a.metric == 'fpr95':
        METRIC, LOWER_BETTER, MNAME = fpr_at_tpr, True, 'FPR95'
    A = load(a.a, a.a_method, a.a_shots)
    B = load(a.b, a.b_method, a.b_shots)
    cats = sorted(set(A) & set(B) - set(a.exclude))
    if not cats:
        raise SystemExit('нет общих категорий: проверьте имена методов (--a-method/--b-method) и число эталонов')
    rng = np.random.default_rng(a.seed)
    data = {}
    for c in cats:
        keys = sorted(set(A[c]) & set(B[c]))
        miss = len(set(A[c]) ^ set(B[c]))
        if miss:
            print(f'  {c}: {miss} снимков есть только у одного метода - исключены')
        lab = np.array([A[c][k][0] for k in keys]); sa = np.array([A[c][k][1] for k in keys]); sb = np.array([B[c][k][1] for k in keys])
        groups = {}
        for i, k in enumerate(keys):
            groups.setdefault((int(lab[i]), strip_id(k) if a.by_strip else k), []).append(i)
        gpos = [np.array(v) for (l, _), v in groups.items() if l == 1]
        gneg = [np.array(v) for (l, _), v in groups.items() if l == 0]
        data[c] = (lab, sa, sb, np.where(lab == 1)[0], np.where(lab == 0)[0], gpos, gneg)
    boot = {c: ([], []) for c in cats}
    for _ in range(a.n):
        for c in cats:
            lab, sa, sb, pos, neg, gpos, gneg = data[c]
            ip = rng.integers(0, len(gpos), len(gpos)); ineg = rng.integers(0, len(gneg), len(gneg))
            idx = np.concatenate([gpos[i] for i in ip] + [gneg[i] for i in ineg])
            boot[c][0].append(METRIC(sa[idx], lab[idx])); boot[c][1].append(METRIC(sb[idx], lab[idx]))
    print(f'\nA = {a.a_method} ({a.a}),  B = {a.b_method} ({a.b}),  перевыборок: {a.n},  режим: ' +
          ('по полосам (кластерный)' if a.by_strip else 'по квадратам (независимые, оптимистично)'))
    print(f'{"категория":12s} {"дефектов/полос":>8s}  {(MNAME+" A [95%]"):>22s}  {(MNAME+" B [95%]"):>22s}  {"A - B [95%]":>24s}  P(A<=B)')
    for c in cats:
        lab, sa, sb, pos, neg, gpos, gneg = data[c]
        ba, bb = np.array(boot[c][0]), np.array(boot[c][1]); diff = ba - bb
        la, ha = ci(ba); lb, hb = ci(bb); ld, hd = ci(diff)
        print(f'{c:12s} {len(pos):4d}/{len(gpos):<3d}  {METRIC(sa, lab):.3f} [{la:.3f}, {ha:.3f}]  {METRIC(sb, lab):.3f} [{lb:.3f}, {hb:.3f}]  '
              f'{METRIC(sa, lab) - METRIC(sb, lab):+.3f} [{ld:+.3f}, {hd:+.3f}]  {(np.mean(diff >= 0) if LOWER_BETTER else np.mean(diff <= 0)):.3f}')
    ma = np.nanmean(np.array([boot[c][0] for c in cats]), axis=0)
    mb = np.nanmean(np.array([boot[c][1] for c in cats]), axis=0)
    pa = np.mean([METRIC(data[c][1], data[c][0]) for c in cats]); pb = np.mean([METRIC(data[c][2], data[c][0]) for c in cats])
    la, ha = ci(ma); lb, hb = ci(mb); ld, hd = ci(ma - mb)
    print(f'{"СРЕДНЕЕ":12s} {"":8s}  {pa:.3f} [{la:.3f}, {ha:.3f}]  {pb:.3f} [{lb:.3f}, {hb:.3f}]  '
          f'{pa - pb:+.3f} [{ld:+.3f}, {hd:+.3f}]  {(np.mean(ma - mb >= 0) if LOWER_BETTER else np.mean(ma - mb <= 0)):.3f}')
    if LOWER_BETTER:
        print('Метрика: доля ложных тревог при полноте 95 % (меньше - лучше); последний столбец: доля перевыборок, где A не лучше B.')
    print('\nКак читать: если интервал «A - B» целиком выше нуля, преимущество A надёжно; если он захватывает ноль - различие в пределах случайности.')


if __name__ == '__main__':
    main()