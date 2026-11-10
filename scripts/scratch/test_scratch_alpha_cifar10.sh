
DATA="CIFAR10"
for seed in 42 #702 703 704
do
#CUDA_VISIBLE_DEVICES=0 python main_scratch.py  --seed $seed --dataset $DATA --PG_alpha 1 --alpha_type "static" & 
CUDA_VISIBLE_DEVICES=3 python main_scratch.py  --seed $seed --dataset $DATA --PG_alpha 0.2 --alpha_type "static" 

done