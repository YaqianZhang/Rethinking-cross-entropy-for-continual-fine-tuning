

#!/bin/bash


for seed in  42 
do

 
    for File4 in  "clrs25.yaml" #"imagenet-r.yaml" #"cub200" "clrs25.yaml" "food101.yaml"
    do


    File4="base_final/"$File4



    OVERRIDES=(
  
        "seed_everything=$seed"
        "module.pet_kwargs.rank=4"
        "module.pet_cls="LoRA""
    )
 

 
  CUDA_VISIBLE_DEVICES=7 python main.py --config=$File4 "${OVERRIDES[@]}" module.PG_alpha=1  \
    module.num_emas=0 module.alpha_type="decrease_sigmoid"  \
    module.loss_type="pg_ce"   





    done
done
    
done
