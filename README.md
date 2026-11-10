# <div align="center"><b> Rethinking cross entropy for continual fine-tuning: policy gradient with entropy annealing </b></div>

<div align="center">

This repository is the official implementation of the paper "Rethinking cross entropy for continual fine-tuning: policy gradient with entropy annealing " (under review).

</div>

## Requirements

To install requirements:

- Install all dependencies via `pip`
    ```shell
    pip install -r requirements.txt
    ```
- Using virtual environment:
    - create a virtual enviornment

    ``` 
    python3 -m venv env_name
    ```
    - activate the virtual environment
    ```
    . env_name/bin/activate
    ```
    - install the required package in the environemtn
    ```
    pip install -r requirements.txt
    ```

### Dataset

1. Create a dataset root diretory, _e.g._, `data`.
2.  `ImageNet-R`,`Food101` and `Cub200` datasets will be automatically downloaded, while [`CLRS` (https://github.com/lehaifeng/CLRS)](https://github.com/lehaifeng/CLRS) requires manual download.
3. Overview of dataset root diretory

    ```shell
   
    data
    └── imagenet-r
        ├── imagenet-r
        ├── train_list.txt
        └── val_list.txt
    └── cub200
        ├──CUB_200_2011
    └── food101
        ├──food-101
    └── clrs25
        ├──CLRS
        ├── airport
        ├── bare-land
        └── beach
        └── ...
    ```



## Training 


- Run code with an experiment config file
    ```shell
    python main.py --config=imagenet-r.yaml
    ```
### Loss functions
- run the baseline cross entropy loss 
    ```
   python main.py --config=imagenet-r.yaml  module.alpha_type="static"  module.PG_alpha=1 module.loss_type="pg_ce" 
   ```
- run the Expected Policy Gradient (EPG) method  
    ```
   python main.py --config=imagenet-r.yaml  module.alpha_type="static"  module.PG_alpha=0 module.loss_type="pg_ce" 
   ```
- run adaptive EPG (aEPG) method
    ```
   python main.py --config=imagenet-r.yaml  module.alpha_type="decrease_sigmoid"  module.loss_type="pg_ce" 
   ```
- run focal loss 
    ```
   python main.py --config=imagenet-r.yaml   module.loss_type="focal" 

   ```
   
### Parameter-efficient-fine-tuning structure (PEFT)

    - By default, the config uses LoRA. To run Adapter or Prefix:


    ```shell
    python main.py --config=imagenet-r.yaml module.pet_cls="Adapter"
    ```
    ```shell
    python main.py --config=imagenet-r.yaml module.pet_cls="Prefix"
    ```
## Evaluation
The evaluation is performend continually during the training in continual learning experiments. The evaluation demo is shown with wandb in offline and anonymous mode. 

To visualize the results in wandb cloud platform,
```      
wandb.init(project=$project_name, mode="online",anonymous="allow",)  # No login required
```
or 

```      
wandb.init(project=$project_name,mode="online",entity="your_wandb_account",)  # wandb login required
```

### Weights & Biases (Wandb) Setup  
1. install wandb 

   - Install Wandb: `pip install wandb`  

2. For Wandb logging:  
   - Log in: `wandb login` (requires an account)  

## Results

Our model achieves the following performance on with a random seed of 42:

### Continual fine-tuning using LoRA on ViT for dataset: Split-ImageNet-R200 with 10 tasks

| Loss name         | End Accuracy | 
| ------------------ |---------------- | 
| CE  |     74.1%        |    
| Focal  |     72.5%        |    
| EPG  |     75.2%        |    
| aEPG  |     76.0%        |    

The script ```scripts/continual_learning/test_example.sh``` is provided to reproduce this result.

### Training from scratch for ResNet-50 using CIFAR100 .

| Loss name         | CIFAR 100 Accuracy | 
| ------------------ |---------------- | 
| alpha=1 (CE)  |       76.9%     |    
| alpha=0.2  |          80.4%   |    

Run the script ```scripts/scratch/test_cifar100.sh``` to reproduce this result.



