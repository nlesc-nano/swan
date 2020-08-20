"""
SCScore taken from: https://github.com/connorcoley/scscore
"""

import gzip
import json
import math

import numpy as np
import pkg_resources
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def get_model_data(name: str) -> str:
    """look for the data for a given model with `name`."""
    path = f"data/scscore/full_reaxys_model_{name}/model.ckpt-10654.as_numpy.json.gz"
    return pkg_resources.resource_filename("swan", path)


class SCScorer():
    def __init__(self, model_name: str, score_scale: int = 5.0, fingerprint_rad: int = 2, fingerprint_len: int = 1024):
        self.score_scale = score_scale
        self._restored = False
        self.fingerprint_len = fingerprint_len
        self.fingerprint_rad = fingerprint_rad
        self.restore(model_name)

    def restore(self, model_name: str):
        weight_path = get_model_data(model_name)
        self._load_vars(weight_path)
        print('Restored variables from {}'.format(weight_path))

    def mol_to_fingerprint(self, mol):
        return np.array(
            AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fingerprint_rad, nBits=self.fingerprint_len, useChirality=True))

    def smile_to_fingerprint(self, smi):
        return self.mol_to_fingerprint(Chem.MolFromSmiles(smi))

    def apply(self, x):
        # Each pair of vars is a weight and bias term
        npairs = len(self.vars)
        for i in range(0, npairs, 2):
            last_layer = (i == npairs - 2)
            w = self.vars[i]
            b = self.vars[i + 1]
            x = np.matmul(x, w) + b
            if not last_layer:
                x = x * (x > 0)  # ReLU
        x = 1 + (self.score_scale - 1) * sigmoid(x)
        return x

    def get_score_from_smi(self, smi: str):
        fp = np.array((self.smile_to_fingerprint(smi)), dtype=np.float32)
        if sum(fp) == 0:
            cur_score = 0.
        else:
            cur_score = self.apply(fp)
        return (smi, cur_score)

    def _load_vars(self, weight_path):
        with gzip.GzipFile(weight_path, 'r') as fin:
            json_bytes = fin.read()  # as UTF-8

        variables = json.loads(json_bytes.decode('utf-8'))
        self.vars = [np.array(x) for x in variables]


if __name__ == '__main__':
    model = SCScorer('1024bool')

    smis = ['CCCOCCC', 'CCCNc1ccccc1']
    for smi in smis:
        (smi, sco) = model.get_score_from_smi(smi)
        print('%.4f <--- %s' % (sco, smi))

    model = SCScorer('2048bool', fingerprint_len=2048)
    smis = ['CCCOCCC', 'CCCNc1ccccc1']
    for smi in smis:
        (smi, sco) = model.get_score_from_smi(smi)
        print('%.4f <--- %s' % (sco, smi))

