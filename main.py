from data.dataloader import TinyImageNetDataLoader
from eval import validate
from train import trainloop
from models.custom_model import CustomNet
import torch
from torch import nn
import random
import wandb
import torch.optim as optim

def main():

  wandb.login(key=)

  num_epochs = input("Insert number of epochs:")

  # Start a new wandb run to track this script.
  run = wandb.init(
    # Set the wandb project where this run will be logged.
    entity="dani-meme",
    project="MLDL_lab3",
    name="Experiment_1",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.001,
        "architecture": "CNN",
        "dataset": "TinyImageNet",
        "epochs": num_epochs,
        "optimizer": "Adam",
        "Batch_size": 64,
    }
  )

  data_loader = TinyImageNetDataLoader(data_dir='dataset/tiny_imagenet/tiny-imagenet-200', batch_size=64)
  train_loader , val_loader = data_loader.get_dataloaders()
  model = CustomNet().cuda()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
  best_acc = 0

  # Run the training process for {num_epochs} epochs
  
  for epoch in range(1, num_epochs + 1):
      print("=================================")
      print("Epoch: ",epoch)

      train_loss,train_acc = trainloop(epoch, model, train_loader, criterion, optimizer)
      # At the end of each training iteration, perform a validation step
      val_loss,val_acc = validate(model, val_loader, criterion)
      run.log({"Train Accuracy":train_acc,"Train Loss":train_loss,"Validation Accuracy":val_acc,"Validation Loss":val_loss})
      # Best validation accuracy
      best_acc = max(best_acc, val_acc)
      

  print(f'Best validation accuracy: {best_acc:.2f}%')
  run.finish()


if __name__ == "__main__":
  main()
