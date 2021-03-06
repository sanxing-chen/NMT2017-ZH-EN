#!/bin/sh

# Heavily borrowed from https://github.com/twairball/fairseq-zh-en
WORKER_NUM=1
TEXT=dataset/
DATADIR=data-bin/wmt17_zh_en

# download, unzip, clean and tokenize dataset.
# pip install jieba
# pip install nltk
python preprocess.py $WORKER_NUM

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
# https://github.com/moses-smt/mosesdecoder
# https://github.com/rsennrich/subword-nmt
MOSESDECODER=../mosesdecoder
$MOSESDECODER/scripts/training/clean-corpus-n.perl $TEXT/train zh en $TEXT/train.clean 3 70
$MOSESDECODER/scripts/training/clean-corpus-n.perl $TEXT/valid zh en $TEXT/valid.clean 3 70

# build subword vocab
SUBWORD_NMT=../subword-nmt/subword_nmt
NUM_OPS=32000

# learn codes and encode separately
CODES=codes.${NUM_OPS}.bpe
echo "Encoding subword with BPE using ops=${NUM_OPS}"
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.clean.en > $TEXT/${CODES}.en
$SUBWORD_NMT/learn_bpe.py -s ${NUM_OPS} < $TEXT/train.clean.zh > $TEXT/${CODES}.zh

echo "Applying vocab to training"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en < $TEXT/train.clean.en > $TEXT/train.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.zh < $TEXT/train.clean.zh > $TEXT/train.${NUM_OPS}.bpe.zh

VOCAB=vocab.${NUM_OPS}.bpe
echo "Generating vocab: ${VOCAB}.en"
cat $TEXT/train.${NUM_OPS}.bpe.en | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.en

echo "Generating vocab: ${VOCAB}.zh"
cat $TEXT/train.${NUM_OPS}.bpe.zh | $SUBWORD_NMT/get_vocab.py > $TEXT/${VOCAB}.zh

# encode validation
echo "Applying vocab to valid"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.en < $TEXT/valid.clean.en > $TEXT/valid.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.zh --vocabulary $TEXT/${VOCAB}.zh < $TEXT/valid.clean.zh > $TEXT/valid.${NUM_OPS}.bpe.zh

# encode test
echo "Applying vocab to test"
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.en --vocabulary $TEXT/${VOCAB}.en < $TEXT/test.en > $TEXT/test.${NUM_OPS}.bpe.en
$SUBWORD_NMT/apply_bpe.py -c $TEXT/${CODES}.zh --vocabulary $TEXT/${VOCAB}.zh < $TEXT/test.zh > $TEXT/test.${NUM_OPS}.bpe.zh

# generate preprocessed data
echo "Preprocessing datasets..."
DATADIR=data-bin/wmt17_zh_en
rm -rf $DATADIR
mkdir -p $DATADIR
fairseq-preprocess --source-lang zh --target-lang en \
    --trainpref $TEXT/train.${NUM_OPS}.bpe --validpref $TEXT/valid.${NUM_OPS}.bpe --testpref $TEXT/test.${NUM_OPS}.bpe \
    --thresholdsrc 0 --thresholdtgt 0 --workers $WORKER_NUM --destdir $DATADIR

# training
# Note that the batch size is also depends on the number of GPUs available,
# the experiment is initially performed in 6 NVIDIA Tesla K40M
echo "Training begins"
mkdir -p checkpoints
fairseq-train $DATADIR \
  -a transformer --optimizer adam -s zh -t en \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-update 200000 \
  --warmup-updates 10000 --warmup-init-lr '1e-7' --lr '0.001' \
  --adam-betas '(0.9, 0.98)' --adam-eps '1e-09' --clip-norm 25.0 \
  --keep-last-epochs 20 --save-dir checkpoints --log-format json > train.log