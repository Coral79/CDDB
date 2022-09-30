# A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials
Pytorch implementation for WACV 2023 paper "A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials".
We propose a continual deepfake detection benchmark (CDDB) over a new collection of deepfakes from
both known and unknown generative models.

**A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials** <br>
Chuqiao Li, Zhiwu Huang, Danda Pani Paudel, Yabin Wang, Mohamad Shahbazi, Xiaopeng Hong, Luc Van Gool <br>
[[Project Page]] [[Paper]](https://arxiv.org/abs/2205.05467.pdf)

## Dependencies
Run `pip install -r requirements.txt` to install required dependencies.

## Datasets
Download the following datasets to `datasets/`
* [CDDB benchmark Dataset]()

Organize the dataset folder as:
```
Project
├── datasets
|   ├── CDDB
|   ├── checkpoints
```



## Training:
### DyTox
Using DyTox, train the model with binary/multi labels, with different sequences:
```
cd DyTox
bash train.sh 0,1 \
    --options configs/data/ganfake_easy.yaml configs/data/ganfake_easy_order.yaml configs/model/ganfake_pretrain_dytox.yaml \
    --name dytox_easy_m1500_sumblog0.1 \
    --data-path ./datasets/CDDB/  \
    --output-basedir ./datasets/checkpoints  \
    --memory-size 1500 \
    --binary_loss sum_b_log \
    --binary_weight 0.1
```

### LUCIR:
Using LUCIR, train the model with binary/multi labels, with different sequences:
```
cd LUCIR
python lucir_main.py --name lucir_easy_m1500_sumasig0.1 --checkpoints_dir ./datasets/checkpoints  --dataroot ./datasets/CDDB/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0 --batch_size 32 --num_epochs 40 --binary_loss sum_a_sig --binary_weight 0.1
```

### iCaRL:
Using iCaRL, train the model with binary/multi labels, with different sequences:
```
cd iCaRL
python main_icarl_CNND.py --name icarl_easy_m1500_sumasig0.1 --checkpoints_dir ./datasets/checkpoints --model_weights ./datasets/checkpoints/no_aug/model_epoch_best.pth --dataroot ./datasets/CDDB/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.1 --binary_loss sum_a_sig
```
### Loss Term:

As described in the paper, we implement diffent kinds of binary loss term

SumLogit
```
--binary_loss sum_a_sig
```
SumFeat
```
--binary_loss sum_b_sig
```
SumLog
```
--binary_loss sum_b_log
```
Max
```
--binary_loss max 
```

## Citation

When using the code/figures/data/etc., please cite our work
```
@inproceedings{li2022continual,
  title={A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials},
  author={Li, Chuqiao and Huang, Zhiwu and Paudel, Danda Pani and Wang, Yabin and Shahbazi, Mohamad and Hong, Xiaopeng and Van Gool, Luc},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2023}
}
```
