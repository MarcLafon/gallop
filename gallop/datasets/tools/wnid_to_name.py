import os
from os.path import dirname, abspath
from nltk.corpus import wordnet as wn

import gallop.lib as lib


datasets_dir = dirname(dirname(abspath(__file__)))

IMAGENET_CLASS_NAMES = lib.load_json(os.path.join(datasets_dir, "files", "imagenet_class_index_clip.json"))


with open(os.path.join(datasets_dir, 'files', 'imagenet21k_wordnet_ids.txt')) as f:
    WORDNET_IDS = [wnid.replace('\n', '') for wnid in f.readlines()]

with open(os.path.join(datasets_dir, 'files', 'imagenet21k_wordnet_lemmas.txt')) as f:
    WORDNET_LEMMAS = [name.replace('\n', '').replace('_', ' ') for name in f.readlines()]

IDS_TO_NAMES = dict(zip(WORDNET_IDS, WORDNET_LEMMAS))


def wnid_to_name(wnid: str, use_nltk: bool = True) -> str:
    if wnid.startswith("XXX_"):
        return wnid[4:]

    if wnid in IMAGENET_CLASS_NAMES.keys():
        return IMAGENET_CLASS_NAMES[wnid]

    if use_nltk:
        synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))
        return synset.name().split('.')[0].replace('_', ' ')
    else:
        return IDS_TO_NAMES[wnid].split(',')[0]
