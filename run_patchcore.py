# -*- coding: utf-8 -*-
"""
PatchCore (anomalib 2.x) на папках, подготовленных make_shifted.py.

Пример:
  python3 run_patchcore.py --root ~/mvtec_shifted/light_1shot --cat carpet grid leather tile wood

Обучение = построение банка признаков из train/good (там ровно k эталонов),
тест = снимки со сдвигом условий. Печатает AUROC по изображению и по пикселям.
Скрипт написан под anomalib 2.x и НЕ проверен на реальном запуске (у автора не было доступа к весам сети):
если упадёт - пришлите текст ошибки целиком.
"""
import argparse, json, os, time


def make_datamodule(root, cat):
    try:
        from anomalib.data import MVTecAD as DM            # anomalib 2.x
    except ImportError:
        from anomalib.data import MVTec as DM              # anomalib 1.x
    kw = dict(root=root, category=cat, train_batch_size=8, eval_batch_size=8, num_workers=0)
    try:
        return DM(**kw, val_split_mode='same_as_test')      # валидация = тест, тестовые снимки не делятся пополам
    except TypeError:
        return DM(**kw)


def _to_list(x):
    if x is None:
        return None
    if hasattr(x, 'detach'):
        x = x.detach().cpu()
    if hasattr(x, 'tolist'):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


def _field(batch, names):
    for n in names:
        if isinstance(batch, dict) and n in batch:
            return batch[n]
        if hasattr(batch, n):
            v = getattr(batch, n)
            if v is not None:
                return v
    return None


def extract_scores(preds):
    """Оценка каждого снимка из результата engine.predict (anomalib 1.x: словари, 2.x: объекты-батчи)."""
    out = []
    for b in preds or []:
        paths = _to_list(_field(b, ['image_path', 'image_paths']))
        scores = _to_list(_field(b, ['pred_score', 'pred_scores', 'anomaly_score']))
        if paths is None or scores is None:
            raise RuntimeError('не нашёл image_path/pred_score; поля батча: ' +
                               str(list(b.keys()) if isinstance(b, dict) else [a for a in dir(b) if not a.startswith('_')]))
        for p, sc in zip(paths, scores):
            sc = sc[0] if isinstance(sc, (list, tuple)) else sc
            d = os.path.basename(os.path.dirname(str(p)))
            out.append({'key': f'{d}/{os.path.basename(str(p))}', 'type': d,
                        'label': int(d != 'good'), 'score': float(sc)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--cat', nargs='+', default=['carpet', 'grid', 'leather', 'tile', 'wood'])
    ap.add_argument('--accelerator', default='cpu', help='cpu (надёжно) или mps (быстрее на Mac, может не поддерживаться)')
    ap.add_argument('--out', default='patchcore_results.json')
    ap.add_argument('--coreset', type=float, default=1.0, help='доля сохраняемых признаков (1.0 - все; у anomalib по умолчанию 0.1)')
    ap.add_argument('--backbone', default=None, help='например wide_resnet50_2 (по умолчанию anomalib)')
    ap.add_argument('--layers', nargs='+', default=None, help='например layer1 layer2 layer3')
    ap.add_argument('--neighbors', type=int, default=None, help='число ближайших соседей (у anomalib по умолчанию 9)')
    ap.add_argument('--save-scores', default='', help='json-файл для оценок каждого снимка')
    ap.add_argument('--tag', default='patchcore', help='имя настройки в файле оценок, например patchcore_A')
    ap.add_argument('--model', default='patchcore', choices=['patchcore', 'padim'], help='модель anomalib')
    a = ap.parse_args()

    from anomalib.engine import Engine
    if a.model == 'padim':
        from anomalib.models import Padim as Patchcore     # тот же код запуска, другая модель
    else:
        from anomalib.models import Patchcore

    root = os.path.expanduser(a.root)
    allres = {}
    score_log = []
    for cat in a.cat:
        t0 = time.time()
        dm = make_datamodule(root, cat)
        import inspect
        supported = set(inspect.signature(Patchcore.__init__).parameters)
        want = {'coreset_sampling_ratio': a.coreset}
        if a.backbone: want['backbone'] = a.backbone
        if a.layers: want['layers'] = a.layers
        if a.neighbors: want['num_neighbors'] = a.neighbors
        kwargs = {k: v for k, v in want.items() if k in supported}
        skipped = sorted(set(want) - set(kwargs))
        print(a.model, 'параметры:', kwargs, ('(не поддерживаются этой версией: %s)' % skipped) if skipped else '')
        model = Patchcore(**kwargs)
        engine = Engine(accelerator=a.accelerator, devices=1)
        engine.fit(model=model, datamodule=dm)
        res = engine.test(model=model, datamodule=dm)
        t_fit_test = time.time() - t0
        try:
            print('    препроцессинг:', getattr(model, 'pre_processor', None).transform, flush=True)
        except Exception:
            pass
        r = res[0] if isinstance(res, list) and res else res
        img = r.get('image_AUROC'); pix = r.get('pixel_AUROC')
        allres[cat] = {'image_AUROC': float(img) if img is not None else None,
                       'pixel_AUROC': float(pix) if pix is not None else None,
                       'all_metrics': {k: float(v) for k, v in r.items()}}
        print(f'=== {cat}: PatchCore  AUROC(изобр.) {img}   AUROC(пиксель) {pix}   ({time.time() - t0:.0f} с)', flush=True)
        if a.save_scores:
            try:
                t1 = time.time()
                preds = engine.predict(model=model, datamodule=dm)
                recs = extract_scores(preds)
                for r_ in recs:
                    r_.update({'category': cat, 'method': a.tag, 'shots': None})
                score_log.extend(recs)
                print(f'    сохранено оценок снимков: {len(recs)} (предсказание {time.time() - t1:.0f} с)', flush=True)
            except Exception as e:
                print('    НЕ УДАЛОСЬ получить оценки снимков:', repr(e)[:400], flush=True)
        score_log.append({'category': cat, 'method': a.tag, 'key': '__time__', 'seconds_total': time.time() - t0, 'seconds_fit_test': t_fit_test,
                          'image_AUROC': allres[cat]['image_AUROC'], 'pixel_AUROC': allres[cat]['pixel_AUROC']})
    vals = [v for v in allres.values() if v['image_AUROC'] is not None]
    if vals:
        mi = sum(v['image_AUROC'] for v in vals) / len(vals)
        mp = [v['pixel_AUROC'] for v in vals if v['pixel_AUROC'] is not None]
        print(f'\n=== Среднее: PatchCore  AUROC(изобр.) {mi:.3f}   AUROC(пиксель) {sum(mp) / len(mp):.3f}' if mp else f'\n=== Среднее: AUROC(изобр.) {mi:.3f}')
    if a.save_scores:
        with open(a.save_scores, 'w') as f:
            json.dump({'source': 'patchcore', 'meta': vars(a), 'records': score_log}, f, ensure_ascii=False)
        print('оценки снимков сохранены в', a.save_scores)
    with open(a.out, 'w') as f:
        json.dump(allres, f, indent=2, ensure_ascii=False)
    print('сохранено в', a.out)


if __name__ == '__main__':
    main()