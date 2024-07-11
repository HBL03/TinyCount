# TinyCount: Efficient Crowd Counting Network for Intelligent Surveillance

This repository contains the code for TinyCount, an efficient crowd counting network designed for intelligent surveillance. The code is based on the C^3 Framework, and we extend our gratitude to the open-source community for their contributions. You can find the original framework [here](https://github.com/gjy3035/C-3-Framework).

## Pre-trained Models and Datasets

The pre-trained TinyCount model and the datasets used (ShanghaiTech, UCF-QNRF, WorldExpo'10) can be downloaded from this [link](https://drive.google.com/drive/folders/1dSrFRwqggoSdfMJMI7tLbdluFIDHOnNF?usp=drive_link). Please organize your directory structure as follows:

```(Your workspace)
├── datasets  
│   ├── SHHA  
│   ├── SHHB  
│   ├── WE  
│   ├── QNRF  
├── TinyCount (this repository)  
│   ├── train.py  
│   ├── test.py  
│   ├── ...
```

## Installation

Ensure you have the correct PyTorch version for your CUDA version and install the required packages using: `pip install -r requirements.txt`.


We have tested the code with the following versions:

- CUDA 12.2
- PyTorch 2.2.1

## Training

Specify the parameters in `config.py`. Additionally, set the dataset path and batch size in `./datasets/XXX/setting.py`. The default values are those used in our experiments. Start training using the command: `python train.py`.

This code supports training for TinyCount, MCNN, CSRNet, and SANet.

## Testing

Download the pre-trained models and save them in the `TinyCount/pth` directory. Set the parameters in `config.py`, including the dataset, network, and pre-trained model path. Start testing using the command: `python test.py`.

This code supports testing for TinyCount, MCNN, and CSRNet.

## Inference Time Measurement

To measure response time on a GPU, use `inference.py`. For CPU measurement, use `inference_cpu.py`. You can specify the model using commands like: `python inference.py --net="TinyCount"`.

Response time measurement is supported for TinyCount, MCNN, CSRNet, and SANet. Recent results are recorded in inference_results.txt and inference_results_cpu.txt.

## Visualization

Visualize the Ground Truth and the model's output density map using: `python visualize.py`.

Make sure to specify `exp_name`, `dataRoot`, and `model_path` in `visualize.py`.

---

We hope this repository helps you in your research and development. For any questions or issues, please feel free to open an issue on this GitHub repository.
