import wandb
from wandb_setup import * 

wandb.login()

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'values': [1e-3, 1e-4, 5e-5]},
        'optimizer': {'values': ["Adam", "AdamW"]},
    }
        
        # U-Net architecture-specific parameters (commented out for now)
        # 'init_features': {'values': [32, 64]},s
        # 'activation': {'values': ["relu", "leaky_relu", "gelu"]},
        # 'dropout': {'values': [0.0, 0.2, 0.5]},
        # 'depth': {'values': [4, 5]},
}

sweep_id = wandb.sweep(sweep_config, project="colon-polyp-segmentation-4")

