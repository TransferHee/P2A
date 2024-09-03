# P2A
This repository contains official PyTorch code implementation for the Prioritized Patch Attack (P2A)

[Asking Model Interperter Where to Attack: Enhancing Adversarial Transferability via Prioritized Patch Attack]

![Overall](https://github.com/TransferHee/P2A/blob/main/Figures/Fig1.png)

## Requirements
- Python == 3.10.8
- pytorch == 1.13.1
- torchvision == 0.14.1
- timm == 0.9.16
- captum == 0.7.0
- numpy == 1.26.4
- pandas == 2.2.1
- pillow == 9.3.0
- tqdm == 4.64.1
  
```bash
pip install -r requirements.txt
```

## Dataset
We borrow the open-source ImageNet subset dataset from [here](https://drive.google.com/drive/folders/1CfobY6i8BfqfWPHL31FKFDipNjqWwAhS).

Please download the data and ensure the directory name of dataset to `./ImageNet1000/...` before run the code.

## Usage

You can simply use our P2A with default parameter setting as follows
```bash
python main.py
```

You can change the source model or other P2A paramter like
```bash
python main.py --model={MODEL_NAME} --layer={LAYER_NAME} --tau={TAU}
```

After you run the above code, the generated adversarial examples would be saved in directory `./result/P2A/{MODEL_NAME}`. Then you can evaluate the performance with `test.py` file
```bash
python test.py --model={MODEL_NAME}
```

## Main Results
![Result](https://github.com/TransferHee/P2A/blob/main/Figures/Result_Table.png)


## Acknowledgement
We referenced a large portion of [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) as our original backbone.
