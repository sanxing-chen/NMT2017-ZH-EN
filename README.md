# WMT 2017 Chinese-English NMT

This project contains pre-processing scripts and Transformer baseline training scripts using [pytorch/fairseq](https://github.com/pytorch/fairseq) for [WMT 2017 Machine Translation of News](http://www.statmt.org/wmt17/translation-task.html) Chinese->English track. The model reaches 20 BLEU on testing dataset, after training for only 2 epochs, while the SOTA result is about 24 BLEU.

# Dataset

In order to run the pre-processing script, we need first download all dataset required in the WMT 2017 MT task, and place them in `./dataset/` folder. Please **remove** all other irrelevant language pairs files for convenience.

# Pre-processing

The pre-processing script followed the steps describes in [Hany Hassan et.al. 2018](https://arxiv.org/pdf/1803.05567.pdf) except the second step. The pre-processing script will get 17.8M bilingual sentence pairs as it's described in the paper. After BPE with 32K merge operations, 50K and 33K will be in the Chinese and English vocabularies separately.

Please modify the `fairseq-preprocess` command in `prepare.sh` to specify the number of cpu workers according to the real situation of your machine.

# Others

## Training monitoring

Refering to this [issue](https://github.com/pytorch/fairseq/issues/227), we provide script `draw_curve.py`. 

## Directory overview
```
.
├── test
│   ├── newstest2017-zhen-ref.en.sgm
│   └── newstest2017-zhen-src.zh.sgm
├── train
│   ├── NJU-newsdev2017-zhen
│   │   ├── newsdev2017-zhen-ref.en.sgm
│   │   ├── newsdev2017-zhen-src.zh.sgm
│   │   └── readme.txt
│   ├── United\ Nations\ Parallel-enzh
│   │   ├── DISCLAIMER
│   │   ├── README
│   │   ├── UNv1.0.en-zh.en
│   │   ├── UNv1.0.en-zh.ids
│   │   ├── UNv1.0.en-zh.zh
│   │   └── UNv1.0.pdf
│   ├── casia2015
│   │   ├── casia2015_ch.txt
│   │   └── casia2015_en.txt
│   ├── casict2011
│   │   ├── casict-A_ch.txt
│   │   ├── casict-A_en.txt
│   │   ├── casict-B_ch.txt
│   │   ├── casict-B_en.txt
│   │   └── readme.txt
│   ├── casict2015
│   │   ├── casict2015_ch.txt
│   │   └── casict2015_en.txt
│   ├── datum2015
│   │   ├── datum_ch.txt
│   │   ├── datum_en.txt
│   │   └── readme.txt
│   ├── datum2017
│   │   ├── Book10_cn.txt
│   │   ├── Book10_en.txt
│   │   ├── Book11_cn.txt
│   │   ├── Book11_en.txt
│   │   ├── Book12_cn.txt
│   │   ├── Book12_en.txt
│   │   ├── Book13_cn.txt
│   │   ├── Book13_en.txt
│   │   ├── Book14_cn.txt
│   │   ├── Book14_en.txt
│   │   ├── Book15_cn.txt
│   │   ├── Book15_en.txt
│   │   ├── Book16_cn.txt
│   │   ├── Book16_en.txt
│   │   ├── Book17_cn.txt
│   │   ├── Book17_en.txt
│   │   ├── Book18_cn.txt
│   │   ├── Book18_en.txt
│   │   ├── Book19_cn.txt
│   │   ├── Book19_en.txt
│   │   ├── Book1_cn.txt
│   │   ├── Book1_en.txt
│   │   ├── Book20_cn.txt
│   │   ├── Book20_en.txt
│   │   ├── Book2_cn.txt
│   │   ├── Book2_en.txt
│   │   ├── Book3_cn.txt
│   │   ├── Book3_en.txt
│   │   ├── Book4_cn.txt
│   │   ├── Book4_en.txt
│   │   ├── Book5_cn.txt
│   │   ├── Book5_en.txt
│   │   ├── Book6_cn.txt
│   │   ├── Book6_en.txt
│   │   ├── Book7_cn.txt
│   │   ├── Book7_en.txt
│   │   ├── Book8_cn.txt
│   │   ├── Book8_en.txt
│   │   ├── Book9_cn.txt
│   │   └── Book9_en.txt
│   ├── neu2017
│   │   ├── NEU_cn.txt
│   │   └── NEU_en.txt
│   └── training-parallel-nc-v13
│       ├── news-commentary-v13.zh-en.en
│       └── news-commentary-v13.zh-en.zh
└── valid
    ├── newsdev2017-zhen-ref.en.sgm
    └── newsdev2017-zhen-src.zh.sgm

12 directories, 70 files
```