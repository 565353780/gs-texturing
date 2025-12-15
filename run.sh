cd ../gstex

CUDA_VISIBLE_DEVICES=3 \
  python train.py \
  -s /home/lichanghao/chLi/Dataset/GS/haizei_1/gs/ \
  -m /home/lichanghao/chLi/Dataset/GS/haizei_1/fitting_gs/ \
  --test_iterations 7000 \
  --save_iterations 7000 \
  --viewer_mode none
