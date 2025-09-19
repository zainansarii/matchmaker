from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from scipy.sparse import csr_matrix

class Indexer:
    def __init__(self):
        self.to_idx: Dict[str, int] = {}
        self.to_id: List[str] = []

    def fit(self, ids: List[str]) -> None:
        for _id in ids:
            if _id not in self.to_idx:
                self.to_idx[_id] = len(self.to_id)
                self.to_id.append(_id)

    def get_idx(self, _id: str) -> int:
        return self.to_idx[_id]

    def __len__(self) -> int:
        return len(self.to_id)
    

@dataclass
class Interactions:
    pairs: List[Tuple[str, str]]

    def build_matrix(self) -> Tuple[csr_matrix, Indexer]:
        users = set()
        for u, v in self.pairs:
            users.add(u)
            users.add(v)
        indexer = Indexer()
        indexer.fit(sorted(users))

        rows, cols, data = [], [], []
        for u, v in self.pairs:
            rows.append(indexer.get_idx(u))
            cols.append(indexer.get_idx(v))
            data.append(1.0)

        n = len(indexer)
        R = csr_matrix((np.array(data), (np.array(rows), np.array(cols))),
                       shape=(n, n), dtype=np.float32)
        R.eliminate_zeros()
        return R, indexer

