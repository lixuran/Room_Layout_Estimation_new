#!/bin/bash
python eval_cuboid.py --dt_glob "tmp_cubic_xy_200_both/*json" --gt_glob "data/test/label_cor/*txt"
python eval_cuboid.py --dt_glob "tmp_cubic_xy_4020_both/*json" --gt_glob "data/test/label_cor/*txt"
python eval_cuboid.py --dt_glob "tmp_cubic_xy_6040_both/*json" --gt_glob "data/test/label_cor/*txt"
python eval_cuboid.py --dt_glob "tmp_cubic_xy_8060_both/*json" --gt_glob "data/test/label_cor/*txt"
python eval_cuboid.py --dt_glob "tmp_cubic_xy_10080_both/*json" --gt_glob "data/test/label_cor/*txt"
python eval_cuboid.py --dt_glob "tmp_cubic_xy_120100_both/*json" --gt_glob "data/test/label_cor/*txt"
python eval_cuboid.py --dt_glob "tmp_cubic_xy_140120_both/*json" --gt_glob "data/test/label_cor/*txt"
python eval_cuboid.py --dt_glob "tmp_cubic_xy_160140_both/*json" --gt_glob "data/test/label_cor/*txt"
