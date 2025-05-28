# Unet-For-Polyp-Segmentation-In-Colonoscopy
Includes a Unet implmentation for polyp segmentation in colonoscopy.

<br>

## Table of Contents

- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Training/Inference code and Performance visualisation](#traininginference-code-and-performance-visualisation)
- [About the Model](#about-the-model)
- [Performance Metrices](#performance-metrics)

<br>

## Installation

To run this project, you need to have Python installed. We recommend using a virtual environment to manage dependencies.

1. **Clone the repository**:
    ```sh
    git clone <repository-url>
    cd <repository-folder>
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

<br>

## Dataset Information
The model used in this project is trained on 'Kvasir-SEG Dataset'. The dataset contains 1000 polyp images and their corresponding ground truth from the Kvasir Dataset v2. The images and its corresponding masks are stored in two separate folders with the same filename. 

- **Source** : [Kvasir SEG Dataset](https://datasets.simula.no/kvasir-seg/)
- **Task** : Polyp Identification in colonoscopy imaged (Identify and mask the polyp from the image).
- **Data** : 1000 captcha polyp images with resolution varying from 332x487 to 1920x1072.
- **Masks** : The images and its corresponding masks are stored in two separate folders with the same filename.

<br>

## Training/Inference code and Performance visualisation

- Refer the notebook for the final training, testing and visualisation : [UNET-FOR-POLYP-SEGMENTATION.ipynb](UNET-FOR-POLYP-SEGMENTATION.ipynb)
- Refert the notebook for the wandb integration and sweep results (only the specific section of code is provided : [WANDB-FOR-POLYP-SEGMENTATION.ipynb](WANDB-FOR-POLYP-SEGMENTATION.ipynb)

<br>

## About the Model
The project uses the Unet model from `segmentation_models_pytorch` for achieving polyp segmentation in colonoscopy images. The best hyperparamets for the model was chosen after running sweeps using wandb and running deliberations on the result. 
The final hyperparameters and other parameters used are :
- encoder_name : efficient-b4
- encoder_weights : imagenet
- in_channels : 3
- classes : 1
- batch_size : 8
- optimizer : AdamW
- learning_rate : 0.0001
- epochs : 11
  
On final testing, the model gave a performance score of 0.8137 (IoU) and 0.8833 (Dice).

<br>

## Performance Metrics
- Training : DiceLoss as loss function.
- Testing  : IoU score and Dice Score as evaluation metrics.
