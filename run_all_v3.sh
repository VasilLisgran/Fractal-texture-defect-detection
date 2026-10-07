#!/bin/bash
# Пакет прогонов v3 после рецензии. Запуск из папки Science_work:
#   caffeinate -i bash run_all_v3.sh
# Ничего не нужно делать во время работы (около 3-4 часов). Компьютер не должен засыпать и не должен быть занят другим.
# Итог: файл v3res/REPORT.txt - его и нужно прислать. Журналы каждого шага: v3res/logs/.
# Если какой-то шаг упадёт, сценарий продолжит остальные; ошибка будет в журнале этого шага.

ARCH=~/aitex/archive          # исходный AITEX
DATA=~/aitex_v3               # куда положить подготовленные данные
RES=v3res                     # куда сохранять оценки и отчёт
CATS="fabric_00 fabric_01 fabric_02 fabric_03 fabric_04 fabric_05 fabric_06"
mkdir -p $RES/logs

run() {  # run <имя журнала> <команда...>
  local name=$1; shift
  echo "[$(date +%H:%M:%S)] $name"
  "$@" > $RES/logs/$name.log 2>&1 || echo "   !!! ошибка, см. $RES/logs/$name.log"
}
OURS="python3 eval_mvtec.py --cat $CATS"
PC="python3 run_patchcore.py --cat $CATS"

# 0. Освещённость полос и позиций
run light python3 measure_light.py --root $ARCH

# 1. Подготовка данных
run prep_s0   python3 prepare_aitex_v3.py --root $ARCH --out $DATA/s0   --shots 1 4 --seed 0
run prep_ctrl python3 prepare_aitex_v3.py --root $ARCH --out $DATA/ctrl --shots 1   --seed 0 --variant ctrl
for s in 1 2 3 4 5; do
  run prep_s$s python3 prepare_aitex_v3.py --root $ARCH --out $DATA/s$s --shots 1 --seed $s
done

# 2. Основной набор: наш метод (полный и облегчённый), простое сравнение, PatchCore A/B, PaDiM
for k in 1 4; do
  run ours_s0_$k  $OURS --root $DATA/s0/${k}shot --shots $k --norm --methods plain affine_iso --mean-w 0.1 --save-scores $RES/ours_s0_$k.json
  run light_s0_$k $OURS --root $DATA/s0/${k}shot --shots $k --norm --methods affine --mean-w 0 --save-scores $RES/light_s0_$k.json
  run pcA_s0_$k   $PC --root $DATA/s0/${k}shot --coreset 1.0 --tag patchcore_A --save-scores $RES/pcA_s0_$k.json --out $RES/pcA_s0_${k}_sum.json
  run pcB_s0_$k   $PC --root $DATA/s0/${k}shot --coreset 0.1 --tag patchcore_B --save-scores $RES/pcB_s0_$k.json --out $RES/pcB_s0_${k}_sum.json
  run padim_s0_$k $PC --root $DATA/s0/${k}shot --model padim --tag padim --save-scores $RES/padim_s0_$k.json --out $RES/padim_s0_${k}_sum.json
done

# 3. Контроль: норма из бездефектных частей дефектных полос
run ours_ctrl_1  $OURS --root $DATA/ctrl/1shot --shots 1 --norm --methods plain affine_iso --mean-w 0.1 --save-scores $RES/ours_ctrl_1.json
run light_ctrl_1 $OURS --root $DATA/ctrl/1shot --shots 1 --norm --methods affine --mean-w 0 --save-scores $RES/light_ctrl_1.json
run pcA_ctrl_1   $PC --root $DATA/ctrl/1shot --coreset 1.0 --tag patchcore_A --save-scores $RES/pcA_ctrl_1.json --out $RES/pcA_ctrl_1_sum.json
run pcB_ctrl_1   $PC --root $DATA/ctrl/1shot --coreset 0.1 --tag patchcore_B --save-scores $RES/pcB_ctrl_1.json --out $RES/pcB_ctrl_1_sum.json

# 4. Абляция по составляющим (облегчённый вариант, быстро) и изометрии без поправки на яркость
S0=$DATA/s0/1shot
run abl_center  $OURS --root $S0 --shots 1 --methods affine --smin 1 --smax 1 --mean-w 0 --save-scores $RES/abl_1_center_only.json
run abl_unb     $OURS --root $S0 --shots 1 --methods affine --smax 0 --mean-w 0 --save-scores $RES/abl_2_fit_unbounded.json
run abl_unbn    $OURS --root $S0 --shots 1 --methods affine --smax 0 --norm --mean-w 0 --save-scores $RES/abl_3_fit_unbounded_norm.json
run abl_bnd     $OURS --root $S0 --shots 1 --methods affine --mean-w 0 --save-scores $RES/abl_4_fit_bounded_nonorm.json
run abl_iso     $OURS --root $S0 --shots 1 --norm --methods affine_iso --mean-w 0 --save-scores $RES/abl_6_iso_no_brightness.json

# 5. Чувствительность к параметрам (не для выбора параметров: они зафиксированы)
run sens_B8    $OURS --root $S0 --shots 1 --norm --methods affine --mean-w 0 --block 8  --save-scores $RES/sens_light_B8.json
run sens_B32   $OURS --root $S0 --shots 1 --norm --methods affine --mean-w 0 --block 32 --save-scores $RES/sens_light_B32.json
run sens_st2   $OURS --root $S0 --shots 1 --norm --methods affine --mean-w 0 --stride 2 --save-scores $RES/sens_light_stride2.json
run sens_st8   $OURS --root $S0 --shots 1 --norm --methods affine --mean-w 0 --stride 8 --save-scores $RES/sens_light_stride8.json
run sens_bw    $OURS --root $S0 --shots 1 --norm --methods affine --mean-w 0 --smin 0.5 --smax 2.0  --save-scores $RES/sens_light_s0.5-2.0.json
run sens_bn    $OURS --root $S0 --shots 1 --norm --methods affine --mean-w 0 --smin 0.8 --smax 1.25 --save-scores $RES/sens_light_s0.8-1.25.json
run sens_w003  $OURS --root $S0 --shots 1 --norm --methods affine_iso --mean-w 0.03 --save-scores $RES/sens_full_w0.03.json
run sens_w03   $OURS --root $S0 --shots 1 --norm --methods affine_iso --mean-w 0.3  --save-scores $RES/sens_full_w0.3.json

# 6. Проверка инвариантности к общему усилению яркости тестовых снимков
for g in 0.7 0.85 1.2 1.4; do
  run gain_light_$g $OURS --root $S0 --shots 1 --norm --methods affine --mean-w 0 --shift gain$g --save-scores $RES/gain_${g}_light.json
  run gain_full_$g  $OURS --root $S0 --shots 1 --norm --methods affine_iso --mean-w 0.1 --shift gain$g --save-scores $RES/gain_${g}_full.json
done

# 7. Разброс по выбору эталона: 5 случайных выборов, 1 эталон
for s in 1 2 3 4 5; do
  run ours_s${s}_1  $OURS --root $DATA/s$s/1shot --shots 1 --norm --methods plain affine_iso --mean-w 0.1 --save-scores $RES/ours_s${s}_1.json
  run light_s${s}_1 $OURS --root $DATA/s$s/1shot --shots 1 --norm --methods affine --mean-w 0 --save-scores $RES/light_s${s}_1.json
  run pcA_s${s}_1   $PC --root $DATA/s$s/1shot --coreset 1.0 --tag patchcore_A --save-scores $RES/pcA_s${s}_1.json --out $RES/pcA_s${s}_1_sum.json
done

# 8. Отчёт: сводка + интервалы (по полосам) для основных сравнений
REP=$RES/REPORT.txt
{
  echo "######## ОСВЕЩЁННОСТЬ ########"; cat $RES/logs/light.log
  echo; echo "######## ПОДГОТОВКА (основной набор и контроль) ########"; grep -h "seed\|ctrl" $RES/logs/prep_s0.log $RES/logs/prep_ctrl.log
  echo; echo "######## СВОДКА ########"; python3 summarize_v3.py --res $RES
  for k in 1 4; do
    for B in pcA:patchcore_A pcB:patchcore_B padim:padim; do
      f=${B%%:*}; m=${B##*:}
      for metric in auroc fpr95; do
        echo; echo "######## ИНТЕРВАЛЫ: наш полный против $m, $k эталон(а), метрика $metric ########"
        [ -f $RES/${f}_s0_$k.json ] && python3 bootstrap_ci.py --by-strip --metric $metric --a $RES/ours_s0_$k.json --a-method affine_iso --a-shots $k \
          --b $RES/${f}_s0_$k.json --b-method $m --exclude fabric_05 2>&1 | grep -v "^$" || echo "   нет файла ${f}_s0_$k.json"
      done
    done
    echo; echo "######## ИНТЕРВАЛЫ: облегчённый против PatchCore A, $k эталон(а) ########"
    [ -f $RES/pcA_s0_$k.json ] && python3 bootstrap_ci.py --by-strip --a $RES/light_s0_$k.json --a-method affine --a-shots $k \
      --b $RES/pcA_s0_$k.json --b-method patchcore_A --exclude fabric_05 2>&1 | grep -v "^$" || echo "   нет файла pcA_s0_$k.json"
  done
  for B in pcA:patchcore_A pcB:patchcore_B; do
    f=${B%%:*}; m=${B##*:}
    echo; echo "######## ИНТЕРВАЛЫ (КОНТРОЛЬ): наш полный против $m, 1 эталон ########"
    [ -f $RES/${f}_ctrl_1.json ] && python3 bootstrap_ci.py --by-strip --a $RES/ours_ctrl_1.json --a-method affine_iso --a-shots 1 \
      --b $RES/${f}_ctrl_1.json --b-method $m --exclude fabric_05 2>&1 | grep -v "^$" || echo "   нет файла ${f}_ctrl_1.json"
  done
  echo; echo "######## ОШИБКИ В ЖУРНАЛАХ (если пусто - всё прошло) ########"
  grep -l -i "traceback\|error" $RES/logs/*.log
} > $REP 2>&1
echo "Готово. Пришлите файл $REP"
