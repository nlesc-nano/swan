"""
SCScore taken from: https://github.com/connorcoley/scscore
"""

import gzip
import json
import math
import pkg_resources

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import six


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_model_data(name: str) -> str:
    """look for the data for a given model with `name`."""
    path = f"data/scscore/full_reaxys_model_{name}/model.ckpt-10654.as_numpy.json.gz"
    return pkg_resources.resource_filename("swan", path)


class SCScorer():
    def __init__(self, score_scale: int = 5.0):
        self.vars = []
        self.score_scale = score_scale
        self._restored = False

    def restore(self, model_name: str, fingerprint_rad: int = 2, fingerprint_len: int = 1024):
        self.fingerprint_len = fingerprint_len
        self.fingerprint_rad = fingerprint_rad
        weight_path = get_model_data(model_name)
        self._load_vars(weight_path)
        print('Restored variables from {}'.format(weight_path))

        if 'uint8' in weight_path or 'counts' in weight_path:
            def mol_to_fp(self, mol):
                # uitnsparsevect
                fp = AllChem.GetMorganFingerprint(mol, self.fingerprint_rad, useChirality=True)
                fp_folded = np.zeros((self.fingerprint_len,), dtype=np.uint8)
                for k, v in six.iteritems(fp.GetNonzeroElements()):
                    fp_folded[k % self.fingerprint_len] += v
                return np.array(fp_folded)
        else:
            def mol_to_fp(self, mol):
                return np.array(AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.fingerprint_rad, nBits=self.fingerprint_len, useChirality=True), dtype=np.bool)
        self.mol_to_fp = mol_to_fp

        self._restored = True
        return self

    def smi_to_fp(self, smi):
        if not smi:
            return np.zeros((self.fingerprint_len,), dtype=np.float32)
        return self.mol_to_fp(self, Chem.MolFromSmiles(smi))

    def apply(self, x):
        if not self._restored:
            raise ValueError('Must restore model weights!')
        # Each pair of vars is a weight and bias term
        for i in range(0, len(self.vars), 2):
            last_layer = (i == len(self.vars)-2)
            W = self.vars[i]
            b = self.vars[i+1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0) # ReLU
        x = 1 + (self.score_scale - 1) * sigmoid(x)
        return x

    def get_score_from_smi(self, smi='', v=False):
        if not smi:
            return ('', 0.)
        fp = np.array((self.smi_to_fp(smi)), dtype=np.float32)
        if sum(fp) == 0:
            if v:
                print('Could not get fingerprint?')
            cur_score = 0.
        else:
            # Run
            cur_score = self.apply(fp)
            if v:
                print('Score: {}'.format(cur_score))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ''
        return (smi, cur_score)

    def _load_vars(self, weight_path):
        with gzip.GzipFile(weight_path, 'r') as fin:    # 4. gzip
            json_bytes = fin.read()                     # 3. bytes (i.e. UTF-8)
            json_str = json_bytes.decode('utf-8')       # 2. string (i.e. JSON)
            self.vars = json.loads(json_str)
            self.vars = [np.array(x) for x in self.vars]


if __name__ == '__main__':
    model = SCScorer()
    model.restore('1024bool')
    smis = ['CCCOCCC', 'CCCNc1ccccc1']
    for smi in smis:
        (smi, sco) = model.get_score_from_smi(smi)
        print('%.4f <--- %s' % (sco, smi))

    # model = SCScorer()
    # model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_2048bool', 'model.ckpt-10654.as_numpy.json.gz'), FP_LEN=2048)
    # smis = ['CCCOCCC', 'CCCNc1ccccc1']
    # for smi in smis:
    #     (smi, sco) = model.get_score_from_smi(smi)
    #     print('%.4f <--- %s' % (sco, smi))

    # model = SCScorer()
    # model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024uint8', 'model.ckpt-10654.as_numpy.json.gz'))
    # smis = ['CCCOCCC', 'CCCNc1ccccc1']
    # for smi in smis:
    #     (smi, sco) = model.get_score_from_smi(smi)
    #     print('%.4f <--- %s' % (sco, smi))
