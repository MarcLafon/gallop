from typing import Type, Dict, Sequence, Union


import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit

import gallop.lib as lib

NoneType = Type[None]


class Subset(Subset):

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
    ) -> NoneType:
        super().__init__(dataset, indices)
        self.targets = self.labels = [dataset.labels[i] for i in indices]

        self.all_names = dataset.all_names
        if hasattr(dataset, 'classes'):
            self.classes = dataset.classes
        if hasattr(dataset, 'wnid_to_idx'):
            self.wnid_to_idx = dataset.wnid_to_idx
        if hasattr(dataset, 'idx_to_wnid'):
            self.idx_to_wnid = dataset.idx_to_wnid
        if hasattr(dataset, 'wnids'):
            self.wnids = [dataset.wnids[i] for i in indices]
        if hasattr(dataset, 'label_names'):
            self.label_names = [dataset.label_names[i] for i in indices]

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str, int]]:
        out = super().__getitem__(idx)
        old_idx = out['index']
        out['index'] = idx
        out['old_index'] = old_idx
        out['old_target'] = out['target']
        out['target'] = self.labels[idx]
        return out


def create_balanced_subset(dts: Dataset, proportion: float = 1., random_state: int = 0) -> Dataset:
    assert isinstance(proportion, float), f"proportion must be a float, got {type(proportion)}."
    assert 0. < proportion <= 1., f"proportion must be in ]0, 1], got {proportion}."

    if proportion == 1.:
        return dts

    labels = torch.tensor([s[-1] for s in dts.samples]).long().view(-1).tolist()

    splits = StratifiedShuffleSplit(n_splits=1, train_size=proportion, random_state=random_state)
    indices, _ = next(splits.split(labels, labels))
    _ = indices.sort()

    return Subset(dts, indices)


def create_few_shots_dataset(dts: Dataset, num_shots: int, seed: int) -> Subset:
    # TODO: Check if remove seed parameter. In therory sklearn uses the global random seed provided by numpy,
    #  so setting the numpy seed should be sufficient
    if num_shots == -1:
        return dts
    if num_shots * len(dts.all_names) > len(dts):
        lib.LOGGER.warning(f"Few shot dataset with {num_shots} shots is larger than the original dataset. ")
        return dts

    labels = dts.labels if not hasattr(dts, '_few_shot_labels') else dts._few_shot_labels

    num_classes = len(set(labels))
    splits = StratifiedShuffleSplit(n_splits=1, train_size=num_shots * num_classes, random_state=seed)
    indices, _ = next(splits.split(labels, labels))
    new_dts = Subset(dts, indices)
    return new_dts
