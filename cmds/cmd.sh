devices=0,1,2,3

CUDA_VISIBLE_DEVICES=${devices} python main.py --model deeplabv3plus_resnet50 --vis_port 28333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --download

# backRazor
devices=4,5,6,7
CUDA_VISIBLE_DEVICES=${devices} python main.py --model uper_res18 --vis_port 28333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.9

devices=0,1,2,3
CUDA_VISIBLE_DEVICES=${devices} python main.py --model uper_res18 --vis_port 28333 --gpu_id 0,1,2,3 --year 2012_aug \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 --backRazorR 0.01
