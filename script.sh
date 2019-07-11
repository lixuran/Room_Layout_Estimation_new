#!/bin/bash
# Basic while loop
python inference.py --flip --pth ckpt/resnet50-rnn.pth --img_glob "data/test/img/*png" --output_dir tmp_cubic_xy_10080_both --max_y_rotate 100 --min_y_rotate 80 --y_rotate_prob 0 --both_rotate True
python inference.py --flip --pth ckpt/resnet50-rnn.pth --img_glob "data/test/img/*png" --output_dir tmp_cubic_xy_120100_both --max_y_rotate 120 --min_y_rotate 100 --y_rotate_prob 0 --both_rotate True
python inference.py --flip --pth ckpt/resnet50-rnn.pth --img_glob "data/test/img/*png" --output_dir tmp_cubic_xy_140120_both --max_y_rotate 140 --min_y_rotate 120 --y_rotate_prob 0 --both_rotate True
python inference.py --flip --pth ckpt/resnet50-rnn.pth --img_glob "data/test/img/*png" --output_dir tmp_cubic_xy_160140_both --max_y_rotate 160 --min_y_rotate 140 --y_rotate_prob 0 --both_rotate True
