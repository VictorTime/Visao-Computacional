
## 🛠Setup

```bash
git clone https://github.com/VictorTime/Visao-Computacional.git
conda create -n ukan python=3.10
conda activate ukan
cd Seg_UKAN && pip install -r requirements.txt
```

**Tips A**: We test the framework using pytorch=1.13.0, and the CUDA compile version=11.6. Other versions should be also fine but not totally ensured.


## 📚Data Preparation
**PANCREAS**:  OS Folds estão aqui [here](https://drive.google.com/file/d/1qhHTyKqmD_XBDoHsO13zgwnMnVVkyY2V/view?usp=drive_link). 

The resulted file structure is as follows.
```
Seg_UKAN
├── inputs
│   ├── busi
│     ├── images
│           ├── malignant (1).png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── malignant (1)_mask.png
|           ├── ...
│   ├── GLAS
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...
│   ├── CVC-ClinicDB
│     ├── images
│           ├── 0.png
|           ├── ...
|     ├── masks
│        ├── 0
│           ├── 0.png
|           ├── ...
```

## 🔖Evaluating Segmentation U-KAN

You can directly evaluate U-KAN from the checkpoint model. Here is an example for quick usage for using our **pre-trained models** in [Segmentation Model Zoo](#segmentation-model-zoo):
1. Download the pre-trained weights and put them to ```{args.output_dir}/{args.name}/model.pth```
2. Run the following scripts to 
```bash
cd Seg_UKAN
python val.py --name ${dataset}_UKAN --output_dir [YOUR_OUTPUT_DIR] 
```

## ⏳Training Segmentation U-KAN

Ah sim, passa o caminho das pastas onde estão o dataset no reorganizar folds unidos, ai tu gera na pasta output
You can simply train U-KAN on a single GPU by specifing the dataset name ```--dataset``` and input size ```--input_size```.
```bash
cd Seg_UKAN
python train.py --arch UKAN --dataset {dataset} --input_w {input_size} --input_h {input_size} --name {dataset}_UKAN  --data_dir [YOUR_DATA_DIR]
```
For example, train U-KAN with the resolution of 256x256 with a single GPU on the BUSI dataset in the ```inputs``` dir:
```bash
cd Seg_UKAN
python train.py --arch UKAN --dataset busi --input_w 256 --input_h 256 --name busi_UKAN  --data_dir ./inputs
```
Please see Seg_UKAN/scripts.sh for more details.
Note that the resolution of glas is 512x512, differing with other datasets (256x256).


Please refer to [Diffusion_UKAN](./Diffusion_UKAN/README.md)

## 🎈Acknowledgements
Greatly appreciate the tremendous effort for the following projects!
- [CKAN](https://github.com/AntonioTepsich/Convolutional-KANs)
--!>
