import sys
from train_val import train
from test import test
from utils import *

def run_sweep_mode():
    from wandb_setup import start_sweep
    from wandb_sweep import sweep_id
    print("Launching wandb sweep:", sweep_id)
    start_sweep(sweep_id, count=5)
    
def run_standard_mode():
    print("Running standard pipeline...")

    model, device, X_test, y_test, train_dice_losses, val_dice_losses = train()

    plot_dice_loss_curves(train_dice_losses, val_dice_losses)

    test(model, device, X_test, y_test)

    visualize_predictions(model, device, X_test, y_test, num_samples_to_plot=5)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py [wandb|normal]")
        sys.exit(1)

    mode = sys.argv[1].lower()
    if mode == "wandb":
        run_sweep_mode()
    elif mode == "normal":
        run_standard_mode()
    else:
        print("Invalid option. Use 'wandb' or 'normal'.")
        sys.exit(1)

