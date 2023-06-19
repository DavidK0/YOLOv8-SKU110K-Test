# This script is in charge of hyperparameter tuning a YOLOv8 model.
# It takes no arguments

from ultralytics import YOLO
import gc # Manual garbage collection is neccesary for hyperparameter tuning
import random
import time

num_trials  = 35 # The number of trials to run
output_file = "hyperparameter_results.txt" # The file to store the search results to
epochs_per_trial = 20

# This function returns a random point in the hyperparameter search space
def get_random_hyperparameters():
    random.seed()
    return {
        "lr0": random.uniform(1e-5, 1e-1), # Initial learning rate
        "lrf": random.uniform(0.01, 1.0), # Final learning rate factor
        "momentum": random.uniform(0.6, 0.98), # Momentum
        "weight_decay": random.uniform(0.0, 0.001), # Weight decay
        "warmup_epochs": random.uniform(0.0, 5.0), # Warmup epochs
        "warmup_momentum": random.uniform(0.0, 0.95), # Warmup momentum
        "box": random.uniform(0.02, 0.2), # Box loss weight,
        "cls": random.uniform(0.2, 4.0) # Class loss weight
    }

if __name__ == "__main__":
    # Overwrite the output file
    with open(output_file, "w") as file:
        pass

    # Keep track of the best hyperparameter values
    best_mAP50 = 0
    best_hyperparameters = None
    best_trial = None

    # Keep track of how long hyperparameter searching takes
    start_time = time.time()

    for trial in range(num_trials):
        # Print a progress bar (large to make it visible)
        print(f"===========================================")
        print(f"============== Progress: {(trial + 1)/num_trials:.0%} ==============")
        print(f"===========================================")

        # Get random hyperparameters
        hyperparameters = get_random_hyperparameters()
        
        # Load a pretrained model
        model = YOLO('yolov8n.pt') # "n" for nano sized

        # Train the model
        model.train(data="SKU500.yaml", epochs=epochs_per_trial,
            # Hyperparameters
            lr0=hyperparameters["lr0"],
            lrf=hyperparameters["lrf"],
            momentum=hyperparameters["momentum"],
            weight_decay=hyperparameters["weight_decay"],
            warmup_epochs=hyperparameters["warmup_epochs"],
            warmup_momentum=hyperparameters["warmup_momentum"],
            box=hyperparameters["box"],
            cls=hyperparameters["cls"]
        )

        # Validate the model
        metrics = model.val()
        mAP50 = metrics.box.map50
        
        # Check if this is the best mAP50 so far
        if mAP50 > best_mAP50:
            best_mAP50 = mAP50
            best_hyperparameters = hyperparameters
            best_trial = trial

        # Output the results of every trial
        print(hyperparameters)
        with open(output_file, "a") as file:
            file.write(f"{trial} {mAP50} {hyperparameters}\n")

        # Delete the model to free up memory
        del model
        gc.collect()

    # Re-output the results of the best trial
    with open(output_file, "a") as file:
        file.write("\nBest Trial\n")
        file.write(f"{best_trial} {best_mAP50} {best_hyperparameters}")

    # Report the final run time
    run_time = time.time() - start_time
    hours, remainder = divmod(run_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Run time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"Average minutes per trial: {run_time/60/num_trials:.1f}")
    print(f"Average seconds per epoch: {run_time/num_trials/epochs_per_trial:.1f}")
