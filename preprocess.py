import jieba
import nltk

import argparse
from pathlib import Path
import re
from multiprocessing import Pool
from functools import partial

DATA_DIR = 'dataset'
TRAIN_DATA_DIR = 'dataset/train'
TRAIN_DATA_ZH_DIR = 'dataset/train.zh'
TRAIN_DATA_EN_DIR = 'dataset/train.en'
VALID_DATA_DIR = 'dataset/valid'
VALID_DATA_ZH_DIR = 'dataset/valid.zh'
VALID_DATA_EN_DIR = 'dataset/valid.en'
TEST_DATA_DIR = 'dataset/test'
TEST_DATA_ZH_DIR = 'dataset/test.zh'
TEST_DATA_EN_DIR = 'dataset/test.en'

LENGTH_RATIO = 1.3

WORKER_NUM = 1

def is_lang(filename: str, lang: str):
    valid_suffix = [f'.{lang}', f'.{lang}.sgm', f'_{lang}.txt']
    has_valid_suffix = False
    for suffix in valid_suffix:
        if filename.endswith(suffix):
            has_valid_suffix = True
    return has_valid_suffix


def is_sgm(filename: str):
    return True if filename.endswith('sgm') else False


def preprocess_text(text: str, is_sgm):

    def strip_line(line: str):
        """Preprocessing to strip tags in SGM files."""

        # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
        if line.startswith("<srcset") or line.startswith("</srcset"):
            return ""
        if line.startswith("<refset") or line.startswith("</refset"):
            return ""
        if line.startswith("<doc") or line.startswith("</doc"):
            return ""
        if line.startswith("<p>") or line.startswith("</p>"):
            return ""
        # Strip <seg> tags.
        line = line.strip()
        if line.startswith("<seg") and line.endswith("</seg>"):
            i = line.index(">")
            return line[i+1:-6]  # Strip first <seg ...> and last </seg>.

    ret = []
    for line in text.split('\n'):
        if is_sgm:
            line = strip_line(line)
        ret.append(line)
    return '\n'.join(ret)


def preprocess_lang(langmarkers, setname):
    line_count_all = 0
    for x in list(Path(DATA_DIR + '/' + setname).glob('*/*')) + list(Path(DATA_DIR + '/' + setname).glob('*')):
        if any(is_lang(x.name, langmarker) for langmarker in langmarkers):
            text = x.read_text().strip()
            text = preprocess_text(text, is_sgm(x.name))
            line_count = len(text.split('\n'))
            line_count_all += line_count
            print(f'{x.parent.name}\t{x.name}\t{line_count}')
            p = Path(DATA_DIR) / f'{setname}.{langmarkers[0]}'
            if not p.exists():
                p.touch()
            p.open(mode='a').write(text + '\n')
    print(langmarkers[0], line_count_all)


def tokenize(line, is_zh):
    tokens = jieba.lcut(line) if is_zh else nltk.word_tokenize(line)
    return ' '.join(tokens), len(tokens)

def contain_chinese(line: str):
    return re.search(u'[\u4e00-\u9fff]', line) != None


all_chars = (chr(i) for i in range(0x110000))
control_chars = ''.join(map(chr, list(range(0,32)) + list(range(127,160))))
control_char_re = re.compile('[%s]' % re.escape(control_chars))

def remove_control_chars(s):
    return control_char_re.sub('', s)

def replace_nonbreaking_whitespace(s):
    return s.replace("\xa0", " ").strip()

def process_line_pair(line_zh, line_en, setname):
    # don't clean testing dataset escept deleting all the empty line
    if setname != 'test':
        line_en = remove_control_chars(line_en)
        line_zh = remove_control_chars(line_zh)

        line_en = replace_nonbreaking_whitespace(line_en)
        line_zh = replace_nonbreaking_whitespace(line_zh)

        if not contain_chinese(line_zh) or contain_chinese(line_en):
            return None

        line_zh, l1 = tokenize(line_zh, True)
        line_en, l2 = tokenize(line_en, False)

        # if l1 > l2 * LENGTH_RATIO or l2 > l1 * LENGTH_RATIO:
        #     continue
    else:
        if len(line_en) == 0 or len(line_zh) == 0:
            return None
        line_zh, l1 = tokenize(line_zh, True)
        line_en, l2 = tokenize(line_en, False)
    return line_zh, line_en

def clean_text(text_zh, text_en, setname):

    cleaned_zh = []
    cleaned_en = []

    text_zh = text_zh.split('\n')
    text_en = text_en.split('\n')

    it = 0
    total_size = len(text_zh)
    with Pool(processes=WORKER_NUM) as pool:
        process_line_pair_fn = partial(process_line_pair, setname=setname)
        line_paiers = list(pool.starmap(process_line_pair_fn, zip(text_zh, text_en)))
        line_paiers = [i for i in line_paiers if i is not None]

    if setname != 'test':
        line_paiers = set(line_paiers)
        line_paiers = list(line_paiers)
    cleaned_zh, cleaned_en = zip(*line_paiers)
    
    Path(DATA_DIR + '/' + setname + '.zh').write_text('\n'.join(cleaned_zh))
    print('cleaned zh', setname, len(cleaned_zh))
    Path(DATA_DIR + '/' + setname + '.en').write_text('\n'.join(cleaned_en))
    print('cleaned en', setname, len(cleaned_en))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess tool for WMT17-ZH-EN data.')
    parser.add_argument('num_workers', metavar='N', type=int, help='number of workers')

    args = parser.parse_args()
    WORKER_NUM = args.num_workers

    preprocess_lang(['zh', 'cn', 'ch'], 'train')
    preprocess_lang(['en'], 'train')

    preprocess_lang(['zh', 'cn', 'ch'], 'valid')
    preprocess_lang(['en'], 'valid')

    preprocess_lang(['zh', 'cn', 'ch'], 'test')
    preprocess_lang(['en'], 'test')

    clean_text(Path(TRAIN_DATA_ZH_DIR).read_text(),
               Path(TRAIN_DATA_EN_DIR).read_text(),
               'train')

    clean_text(Path(VALID_DATA_ZH_DIR).read_text(),
               Path(VALID_DATA_EN_DIR).read_text(),
               'valid')

    clean_text(Path(TEST_DATA_ZH_DIR).read_text(),
               Path(TEST_DATA_EN_DIR).read_text(),
               'test')
