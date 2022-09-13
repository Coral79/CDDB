# A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials
Pytorch implementation for 2023 WACV paper "A Continual Deepfake Detection Benchmark: Dataset, Methods, and Essentials".
We propose a new a continual deepfake detection benchmark (CDDB) over a new collection of deepfakes from
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
python DyTox/train.py ...
```

### LUCIR:
Using LUCIR, train the model with binary/multi labels, with different sequences:
```
python LUCIR/train.py ...
```

### iCaRL:
Using iCaRL, train the model with binary/multi labels, with different sequences:
```
python iCaRL/train.py ...
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
