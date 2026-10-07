#!/bin/bash
RES=v3res
REP=$RES/REPORT_core.txt
cmp() {
  if [ -f "$2" ] && [ -f "$5" ]; then
    echo; echo "######## $1 ########"
    python3 bootstrap_ci.py --by-strip --metric $7 --a $2 --a-method $3 --a-shots $4 --b $5 --b-method $6 --exclude fabric_05 2>&1 | grep -v "^$"
  fi
}
{
  echo "######## ОСВЕЩЁННОСТЬ ########"; cat $RES/logs/light.log 2>/dev/null
  echo; echo "######## ПОДГОТОВКА (основной набор и контроль) ########"; grep -h "seed" $RES/logs/prep_s0.log $RES/logs/prep_ctrl.log 2>/dev/null
  echo; echo "######## СВОДКА ########"; python3 summarize_v3.py --res $RES
  for k in 1 4; do
    for B in pcA:patchcore_A pcB:patchcore_B padim:padim; do
      f=${B%%:*}; m=${B##*:}
      for metric in auroc fpr95; do
        cmp "ИНТЕРВАЛЫ: наш полный против $m, $k эталон(а), $metric" $RES/ours_s0_$k.json affine_iso $k $RES/${f}_s0_$k.json $m $metric
      done
    done
    cmp "ИНТЕРВАЛЫ: облегчённый против patchcore_A, $k эталон(а), auroc" $RES/light_s0_$k.json affine $k $RES/pcA_s0_$k.json patchcore_A auroc
  done
  for B in pcA:patchcore_A pcB:patchcore_B; do
    f=${B%%:*}; m=${B##*:}
    cmp "ИНТЕРВАЛЫ (КОНТРОЛЬ): наш полный против $m, 1 эталон, auroc" $RES/ours_ctrl_1.json affine_iso 1 $RES/${f}_ctrl_1.json $m auroc
  done
  echo; echo "######## ФАЙЛЫ В v3res ########"; ls $RES | grep json | tr '\n' ' '
  echo; echo; echo "######## ЖУРНАЛЫ С ОШИБКАМИ (если пусто - всё прошло) ########"
  grep -l -i "traceback\|error" $RES/logs/*.log 2>/dev/null
} > $REP 2>&1
echo "Готово. Пришлите файл $REP"
