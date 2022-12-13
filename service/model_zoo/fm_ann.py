# pylint: disable=R0902
import os
import pickle
from typing import Any, Dict, List, Tuple

import nmslib
import numpy as np
import pandas as pd


class FmAnn(object):
    """ANN model on top of user and item embeddings."""
    attrs_to_load = [
        'user_embeddings',
        'item_embeddings',
        'interactions',
    ]

    def __init__(
        self,
        dirname: str,
        ef_construction: int = 256,
        n_threads_construction: int = 4,
        ef_search: int = 256,
        m: int = 72,
        user_column: str = 'user_id',
        item_column: str = 'item_id',
    ) -> None:
        """Initialize an ANN model.

        Args:
            dirname: Directory with fixtures.
            ef_construction: The size of the dynamic list
                for the nearest neighbors during index construction.
            n_threads_construction: Index construction parameter.
            ef_search: The size of the dynamic list
                for the nearest neighbors during search.
            m: The number of bidirectional links.
            user_column: The user column name.
            item_column: The item column name.
        """
        self.user_column = user_column
        self.item_column = item_column
        self.index = nmslib.init(
            method='hnsw',
            space='negdotprod',
            data_type=nmslib.DataType.DENSE_VECTOR,
        )
        state = self._load(dirname)
        self.user_embeddings = self._aug_zero(
            state['user_embeddings'],
        )
        self.item_embeddings = self._aug_inner_product(
            state['item_embeddings'],
        )
        self.interactions = state['interactions']
        self._watched = self._build_watched(self.interactions)
        mappings = self._build_mappings(self.interactions)
        self.e2i_user_ids, self.i2e_user_ids = mappings[0], mappings[1]
        self.e2i_item_ids, self.i2e_item_ids = mappings[2], mappings[3]
        self._create_index(
            self.item_embeddings,
            ef_construction=ef_construction,
            n_threads=n_threads_construction,
            ef_search=ef_search,
            m=m,
        )
        # Рассчитаем средний embedding
        self.mean_user = np.mean(self.user_embeddings, axis=0)

    def predict(self, user_id: int, k_recs: int = 10) -> List[int]:
        """Get prediction for the given user.

        Args:
            user_id: The user ID.
            k_recs: The number of recommendations to return.

        Returns:
            The list with the recommendations (len == k_recs).
        """
        internal_user_id = self.e2i_user_ids.get(user_id)
        if internal_user_id is not None:
            user_embedding = self.user_embeddings[internal_user_id]
        else:
            # Для холодных пользователей используем среднее значение
            user_embedding = self.mean_user
        recs = self.index.knnQuery(user_embedding, k=100)[0]
        recs = np.array(list(map(self.i2e_item_ids.get, recs)))
        if internal_user_id is not None:
            watched = np.array(self._watched.loc[user_id])
            recs = recs[np.isin(recs, watched, invert=True)]
        return recs[:k_recs].tolist()

    def _build_watched(self, interactions: pd.DataFrame) -> pd.Series:
        user_groups = interactions.groupby(self.user_column)
        return user_groups[self.item_column].apply(list)

    def _build_mappings(
        self, interactions: pd.DataFrame,
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        external_user_ids = interactions[self.user_column].sort_values(
        ).unique()
        external_item_ids = interactions[self.item_column].sort_values(
        ).unique()
        e2i_user_ids, i2e_user_ids = self._build_mapping(
            external_user_ids,
        )
        e2i_item_ids, i2e_item_ids = self._build_mapping(
            external_item_ids,
        )
        return e2i_user_ids, i2e_user_ids, e2i_item_ids, i2e_item_ids

    def _build_mapping(self, ids: np.ndarray) -> Tuple[Dict, Dict]:
        e2i = {e_id: i_id for i_id, e_id in enumerate(ids)}
        i2e = {i_id: e_id for e_id, i_id in e2i.items()}
        return e2i, i2e

    def _load(
        self, dirname: str,
    ) -> Dict[str, Any]:
        attr_values = {}
        for attr_name in self.attrs_to_load:
            attr_filename = os.path.join(
                dirname, '{0}.pickle'.format(attr_name),
            )
            with open(attr_filename, 'rb') as attr_file:
                attr_values[attr_name] = pickle.load(attr_file)
        return attr_values

    def _create_index(
        self,
        embeddings: np.ndarray,
        ef_construction: int,
        n_threads: int,
        ef_search: int,
        m: int,
    ) -> None:
        index_time_params = {
            'M': m,
            'indexThreadQty': n_threads,
            'efConstruction': ef_construction,
        }
        aug_item_embeddings = self._aug_inner_product(embeddings)
        self.index.addDataPointBatch(aug_item_embeddings)
        self.index.createIndex(index_time_params)
        query_time_params = {'efSearch': ef_search}
        self.index.setQueryTimeParams(query_time_params)

    def _aug_inner_product(self, factors: np.ndarray) -> np.ndarray:
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = np.max(normed_factors)
        extra_dim = np.sqrt(
            max_norm ** 2 - normed_factors ** 2,
        ).reshape(-1, 1)
        return np.append(factors, extra_dim, axis=1)

    def _aug_zero(self, factors: np.ndarray) -> np.ndarray:
        zero = np.zeros((factors.shape[0], 1))
        return np.append(factors, zero, axis=1)
