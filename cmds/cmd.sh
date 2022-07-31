devices=6,7

CUDA_VISIBLE_DEVICES=${devices} python main.py --model deeplabv3plus_mobilenet --vis_port 28333 --gpu_id 0,1 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16

# backRazor
devices=2,3
CUDA_VISIBLE_DEVICES=${devices} python main.py --model deeplabv3plus_mobilenet --vis_port 28333 --gpu_id 0,1 --year 2012_aug \
--crop_val --lr 0.005 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.9
