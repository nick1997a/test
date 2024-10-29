import wandb
import random
wandb.login(key='e037589a673b1270f1a7e4ad5504c331f92b01be')
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="crossencoder",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "DAA",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()