#!/bin/bash
# Доверительные интервалы, которых не хватает в статье. Всё считается по УЖЕ сохранённым
# оценкам из папки v3res: модели не перезапускаются, занимает секунды.
# Запуск из Science_work:  bash optional_intervals.sh > intervals.txt
RES=v3res
B="python3 bootstrap_ci.py --by-strip --exclude fabric_05"

echo "######## наш полный против простого сравнения (1 эталон) ########"
$B --a $RES/ours_s0_1.json --a-method affine_iso --a-shots 1 --b $RES/ours_s0_1.json --b-method plain --b-shots 1
echo; echo "######## наш полный против простого сравнения (4 эталона) ########"
$B --a $RES/ours_s0_4.json --a-method affine_iso --a-shots 4 --b $RES/ours_s0_4.json --b-method plain --b-shots 4
echo; echo "######## облегчённый против простого сравнения (1 эталон) ########"
$B --a $RES/light_s0_1.json --a-method affine --a-shots 1 --b $RES/ours_s0_1.json --b-method plain --b-shots 1

# Абляция: каждая ступень против «только центрирование» и против облегчённого варианта
echo; echo "######## абляция: только центрирование против простого сравнения ########"
$B --a $RES/abl_1_center_only.json --a-method affine --a-shots 1 --b $RES/ours_s0_1.json --b-method plain --b-shots 1
echo; echo "######## абляция: облегчённый (границы+нормировка) против «только центрирование» ########"
$B --a $RES/light_s0_1.json --a-method affine --a-shots 1 --b $RES/abl_1_center_only.json --b-method affine --b-shots 1
echo; echo "######## абляция: подгонка без границ и нормировки против «только центрирование» ########"
$B --a $RES/abl_2_fit_unbounded.json --a-method affine --a-shots 1 --b $RES/abl_1_center_only.json --b-method affine --b-shots 1
echo; echo "######## абляция: подгонка с границами без нормировки против «только центрирование» ########"
$B --a $RES/abl_4_fit_bounded_nonorm.json --a-method affine --a-shots 1 --b $RES/abl_1_center_only.json --b-method affine --b-shots 1
echo; echo "######## абляция: полный метод против облегчённого (вклад изометрий и поправки на яркость) ########"
$B --a $RES/ours_s0_1.json --a-method affine_iso --a-shots 1 --b $RES/light_s0_1.json --b-method affine --b-shots 1
