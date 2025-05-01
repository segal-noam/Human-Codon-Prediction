import wandb
import random


# Login to W&B
wandb.login(key="fb87cc95d8c69efe715ee816e61954c9776c372d")

# Replace with the path to your W&B run directory
run_directory = "./wandb/run-directory"

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
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


# Save run locally
wandb.save(run_directory)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()
