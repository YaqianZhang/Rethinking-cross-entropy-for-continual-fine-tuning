import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torchvision.models import resnet50
import time
import wandb
from loss import PG_CELoss, CELoss,get_adaptive_alpha,Symmetric_KL_Loss

import numpy as np
import random

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 256
    momentum = 0.9
    initial_lr = 0.1
    epochs = 350  # 150 + 100 + 100
    milestones = [150, 225]  # When to decrease learning rate
    gamma = 0.1  # Factor to decrease learning rate by

    optm = "SGD"

    params = {
        "loss_type": args.loss_type,
        "PG_alpha": args.PG_alpha,
        "kl_A":args.kl_A,
        "alpha_min":args.alpha_min,
        "alpha_max":args.alpha_max,
        "beta":args.beta,
        "alpha_type": args.alpha_type,
        "batch_size": batch_size,
        "optimizer": optm,
        "epochs": epochs,
        "initial_lr": initial_lr,
        "milestones": milestones,
        "gamma": gamma,
        "seed": args.seed,
        "dataset": args.dataset
    }

    wandb.init(project="public_demo_"+args.dataset, 
               anonymous="allow",
               config=params,
                       mode="online",
               )  # No login required

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalization for validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load dataset
    dataset=args.dataset
    if dataset == "CIFAR10":
        num_classes = 10
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    elif dataset == "CIFAR100":
        num_classes = 100
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError("Dataset must be either CIFAR10 or CIFAR100")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Set random seed for reproducibility
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    # Modify ResNet-50 for CIFAR (32x32 images)
    class ResNet50CIFAR(nn.Module):
        def __init__(self, num_classes=num_classes):
            super(ResNet50CIFAR, self).__init__()
            # Load pre-trained ResNet-50
            self.model = resnet50(pretrained=False)
            # Modify the first convolution layer for CIFAR (3x32x32)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove the max pooling layer
            self.model.maxpool = nn.Identity()
            # Modify the final fully connected layer
            self.model.fc = nn.Linear(2048, num_classes)
            
        def forward(self, x):
            return self.model(x)

    # Initialize the model
    model = ResNet50CIFAR().to(device)

    # Loss function
    if params["loss_type"] == "ce":

        criterion =CELoss( beta = params["beta"],
                                    temperature = 1
        )
    elif params["loss_type"] == "symmetric_kl":
        criterion=Symmetric_KL_Loss(A=params["kl_A"],
                                        temperature = 1,
                                        )
    else:
        
        criterion = PG_CELoss(PG_alpha=args.PG_alpha, temperature=1.0)
    if optm == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=5e-4)
    elif optm == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=initial_lr,)
    else:
        raise NotImplementedError

    # Learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # Training function
    def train(epoch, current_gradient_step):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        total_gradient_step = epochs * len(train_loader)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            if params["alpha_type"] != "static" and hasattr(criterion, "alpha"):  
                # criterion.alpha = get_adaptive_alpha(batch_idx, len(train_loader),
                #                         type=params["alpha_type"],
                #                         )
                alpha01 = get_adaptive_alpha(current_gradient_step,total_gradient_step,
                                    type=params["alpha_type"],
                                    )
                change = params["alpha_max"]-params["alpha_min"]
                criterion.alpha = params["alpha_min"] + change*alpha01
            
            loss, loss_all, pre_entropy, post_entropy = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            current_gradient_step += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} '
                      f'| Loss: {loss.item():.3f} | Acc: {100.*correct/total:.3f}%')

            if batch_idx % 10 == 0:
                wandb.log({"gradient_step": current_gradient_step,
                           "incoming_acc": 100.*correct/total,
                           "loss": loss.item(),
                           "pre_entropy": pre_entropy.item(),
                           "alpha": criterion.alpha,
                           })
        
        return train_loss/(batch_idx+1), 100.*correct/total, current_gradient_step

    # Testing function
    def test(epoch):
        model.eval()
        test_loss = 0
        correct = 0
        top5_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss, _, _, _ = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_correct += top5_pred.eq(targets.view(-1, 1)).sum().item()
                
        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        top5_accuracy = 100. * top5_correct / total
        print(f'Test Epoch: {epoch} | Loss: {test_loss:.3f} | Acc: {accuracy:.3f}%')
        return test_loss, accuracy,top5_accuracy

    set_seed(args.seed)
    # Training loop
    best_acc = 0
    gradient_step = 0
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss, train_acc, gradient_step = train(epoch, gradient_step)
        test_loss, test_acc , top5_test_acc= test(epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f'resnet50_{dataset.lower()}_best.pth')
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch} completed in {epoch_time:.2f}s | '
              f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)')
        print(f'Current LR: {scheduler.get_last_lr()[0]:.6f}')
        wandb.log({"epoch": epoch,
                   "train_loss": train_loss,
                   "test_loss": test_loss,
                   "train_acc": train_acc,
                   "test_acc": test_acc,
                   "top5_test_acc": top5_test_acc, 
                   "best_acc": best_acc,
                   "current_lr": scheduler.get_last_lr()[0]
                   })

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"], help="Dataset to use")
    parser.add_argument("--PG_alpha", type=float, default=1, help="Initial value for PG alpha")
    parser.add_argument("--seed", type=int, default=0, help="Initial value for PG alpha")
    parser.add_argument("--alpha_type", type=str, default="static", help="Type of alpha adaptation")
    parser.add_argument("--loss_type", type=str, default="pg_ce", help="Type of alpha adaptation")
    parser.add_argument("--beta", type=float, default=0, help="Type of alpha adaptation")
    parser.add_argument("--kl_A", type=float, default=8, help="Type of alpha adaptation")
    parser.add_argument("--alpha_min", type=float, default=0, help="Type of alpha adaptation")
    parser.add_argument("--alpha_max", type=float, default=1, help="Type of alpha adaptation")
    args = parser.parse_args()
    
    # main(dataset=args.dataset, 
    #      PG_alpha=args.PG_alpha, 
    #      alpha_type=args.alpha_type,
    #      seed=args.seed,
    #      loss_type = args.loss_type,
    #      beta = args.beta,
    #      kl_A = args.kl_A)
    main(args)