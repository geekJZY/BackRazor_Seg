devices=0,1,2,3

CUDA_VISIBLE_DEVICES=${devices} python main.py --model deeplabv3_resnet50 --vis_port 28333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.001


CUDA_VISIBLE_DEVICES=${devices} python main.py --model deeplabv3_resnet50 --vis_port 28333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.8

# backRazor
devices=4,5,6,7
CUDA_VISIBLE_DEVICES=${devices} python main.py --model uper_res18 --vis_port 28333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.9

devices=4,5,6,7
CUDA_VISIBLE_DEVICES=${devices} python main.py --model uper_res18 --vis_port 28333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.01


# mobile no_backRazor
CUDA_VISIBLE_DEVICES=${devices} python main.py --model deeplabv3_mobilenet --vis_port 23333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16

# mobile prune 0.0001%
CUDA_VISIBLE_DEVICES=${devices} python main.py --model deeplabv3_mobilenet --vis_port 24333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.1

# mobile prune 10%
devices=4,5,6,7
CUDA_VISIBLE_DEVICES=${devices} python main.py --model deeplabv3_mobilenet --vis_port 28333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.1

# mobile prune 10%, single card
CUDA_VISIBLE_DEVICES=4 python main.py --model deeplabv3_mobilenet --vis_port 23333 --gpu_id 0 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16

CUDA_VISIBLE_DEVICES=5 python main.py --model deeplabv3_mobilenet --vis_port 23633 --gpu_id 0 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.1

python main.py --model deeplabv3_mobilenet --vis_port 23634 --gpu_id 6 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.2

python main.py --model deeplabv3_mobilenet --vis_port 23635 --gpu_id 7 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.5

python main.py --model deeplabv3_mobilenet --vis_port 23632 --gpu_id 1 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.7

CUDA_VISIBLE_DEVICES=0 python main.py --model deeplabv3_mobilenet --vis_port 23636 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.9

CUDA_VISIBLE_DEVICES=3 python main.py --model deeplabv3_mobilenet --vis_port 23636 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.95

CUDA_VISIBLE_DEVICES=4 python main.py --model deeplabv3_mobilenet --vis_port 23638 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.97


# four cards ablation
python main.py --model deeplabv3_mobilenet --vis_port 21633 --gpu_id 1,2,3,5 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --name_append 4cards


CUDA_VISIBLE_DEVICES=0 python main.py --model deeplabv3_mobilenet --vis_port 23636 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16
