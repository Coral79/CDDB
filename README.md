# A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials
Pytorch implementation for WACV 2023 paper "A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials".
We propose a continual deepfake detection benchmark (CDDB) over a new collection of deepfakes from
both known and unknown generative models.

**A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials** <br>
Chuqiao Li, Zhiwu Huang, Danda Pani Paudel, Yabin Wang, Mohamad Shahbazi, Xiaopeng Hong, Luc Van Gool <br>
[[Project Page]](https://arxiv.org/abs/2205.05467.pdf) [[Paper]](https://arxiv.org/abs/2205.05467.pdf) [[Video]](https://www.youtube.com/watch?v=bszy34vY-2o)

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
```



## Training:
### DyTox
Using DyTox, train the model with binary/multi labels, with different sequences:
```
cd DyTox
bash train.sh 0,1 \
    --options configs/data/ganfake_easy.yaml configs/data/ganfake_easy_order.yaml configs/model/ganfake_pretrain_dytox.yaml \
    --name dytox_ganfake_easy_m1500_sumblog0.01 \
    --data-path ./datasets/CDDB/  \
    --output-basedir ./checkpoints  \
    --memory-size 1500 \
    --binary_loss sum_b_log \
    --binary_weight 0.01
```

### LUCIR:
Using LUCIR, train the model with binary/multi labels, with different sequences:
```
python lucir_main.py --name icarl_df --checkpoints_dir ./checkpoints  --dataroot ./datasets/CDDB/ --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0 --batch_size 32 --num_epochs 40 --binary_loss sum_a_sig --binary_weight 0.1
```

### iCaRL:
Using iCaRL, train the model with binary/multi labels, with different sequences:
```
python main_icarl_CNND.py --name MT_7_bm_sum_b_sig01_15 --checkpoints_dir ./checkpoints --model_weights /home/wangyabin/workspace/datasets/DeepFake_Data/checkpoints/no_aug/model_epoch_best.pth --dataroot /home/wangyabin/workspace/datasets/DeepFake_Data/release --task_name gaugan,biggan,cyclegan,imle,deepfake,crn,wild --multiclass  0 0 1 0 0 0 0  --batch_size 32 --num_epochs 30 --schedule 10 20 30 --add_binary --binary_weight 0.1 --binary_loss sum_b_sig
```

## Citation

When using the code/figures/data/etc., please cite our work
```
@article{li2022continual,
  title={A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials},
  author={Li, Chuqiao and Huang, Zhiwu and Paudel, Danda Pani and Wang, Yabin and Shahbazi, Mohamad and Hong, Xiaopeng and Van Gool, Luc},
  journal={arXiv preprint arXiv:2205.05467},
  year={2022}
}
```
