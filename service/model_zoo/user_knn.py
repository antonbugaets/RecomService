# pylint: disable=R0902
import os
import pickle
from collections import Counter
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import numpy as np
import pandas as pd
import scipy as sp
from implicit.nearest_neighbours import ItemItemRecommender

T = TypeVar('T', bound='UserKnn')
Mapping = Dict[int, int]


class UserKnn(object):
    """User KNN model."""
    attrs_to_save = [
        'user_knn',
        'k_users',
        'cold_user_threshold',
        'cold_item_threshold',
        'popular',
        'user_column',
        'item_column',
        'weight_column',
        'ranking',
        'interactions',
    ]

    def __init__(
        self,
        user_knn: ItemItemRecommender,
        k_users: int,
        cold_user_threshold: int,
        cold_item_threshold: int,
        popular: pd.DataFrame,
        user_column: str = 'user_id',
        item_column: str = 'item_id',
        weight_column: Optional[str] = None,
        ranking: str = 'max',
    ):
        """Create user KNN model.

        Args:
            user_knn: Base user knn to use.
            k_users: KNN parameter.
            cold_user_threshold: Number of interactions threshold for users.
            cold_item_threshold: Number of interactions threshold for items.
            popular: List with popular items.
            user_column: User column in dataset.
            item_column: Item column in dataset.
            weight_column: Weight column in dataset.
            ranking: Agg function to user for item scores.
        """
        self.user_knn = user_knn
        self.k_users = k_users
        self.cold_user_threshold = cold_user_threshold
        self.cold_item_threshold = cold_item_threshold
        self.popular = popular
        self.user_column = user_column
        self.item_column = item_column
        self.weight_column = weight_column
        self.ranking = ranking
        self.interactions = pd.DataFrame()
        self._watched = pd.Series(dtype=int)
        self._item_idf = pd.DataFrame()
        self._users_mapping: Mapping = {}
        self._users_inv_mapping: Mapping = {}
        self._items_mapping: Mapping = {}
        self._items_inv_mapping: Mapping = {}
        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame) -> None:
        """Train model.

        Args:
            interactions: Dataset for training on.
        """
        # Удаляем холодные записи.
        interactions = self._filter(interactions)
        self._preprocess(interactions)
        interaction_matrix = self._get_interaction_matrix(
            self.interactions,
        )

        self.user_knn.fit(interaction_matrix)
        self.is_fitted = True

    def predict(
        self, user_id: int, k_recs: int = 10,
    ) -> List[int]:
        """Predict recommendations for user with given ID.

        Args:
            user_id: User ID.
            k_recs: Number of recommendations.

        Returns:
            List of item IDs.
        """
        if not self.is_fitted:
            raise ValueError(
                'Model not fitted, call fit before predicting.',
            )
        user_watched = []
        if user_id in self._watched:
            user_watched = self._watched.loc[user_id]
        if len(user_watched) >= self.cold_user_threshold:
            similar_user_ids, scores = self._get_similar_users(
                user_id, self.k_users,
            )
            recs = pd.DataFrame(
                {
                    self.user_column: similar_user_ids,
                    self.item_column: self._watched.loc[
                        similar_user_ids].values,
                    'score': scores,
                },
            )

            recs = recs.explode(self.item_column)
            # Применяем аггрегирующую функцию к скорам
            recs['agg_score'] = recs.groupby(
                self.item_column,
            )['score'].transform(self.ranking)
            recs = recs.sort_values(
                'agg_score', ascending=False,
            ).drop_duplicates(
                self.item_column,
            ).merge(
                self._item_idf,
                left_on=self.item_column,
                right_on='index',
                how='left',
            )
            # Домножаем на значение IDF
            recs['agg_score'] *= recs['idf']
            recs.sort_values(
                'agg_score', ascending=False, inplace=True,
            )
            # Дополняем рекомендации популярным
            recs = pd.concat([recs, self.popular]).drop_duplicates(
                self.item_column,
            )
            # Убираем из списка то, что пользователь уже смотрел
            mask = recs[self.item_column].isin(user_watched)
            recs = recs[~mask]
        else:
            # Для холодных пользователей рекомендуем популярное
            # Убираем из списка то, что пользователь уже смотрел
            mask = self.popular[self.item_column].isin(user_watched)
            recs = self.popular[~mask]
        recs = recs.copy()
        recs['rank'] = np.arange(1, len(recs) + 1)
        return recs[recs['rank'] <= k_recs][self.item_column].tolist()

    @classmethod
    def load(cls: Type[T], dirpath: str) -> T:
        """Load model state.

        Args:
            dirpath: Path to the dir with the saved model.

        Returns:
            Loaded model.
        """
        attrs_dict = {}
        for attr_name in cls.attrs_to_save:
            attr_filepath = os.path.join(
                dirpath, '{0}.pickle'.format(attr_name),
            )
            with open(attr_filepath, 'rb') as attr_file:
                attr_value = pickle.load(attr_file)
            attrs_dict[attr_name] = attr_value
        interactions = attrs_dict.pop('interactions')
        model = cls(**attrs_dict)
        model._preprocess(interactions)
        model.is_fitted = True
        return model

    def save(self, dirpath: str) -> None:
        """Save model state.

        Args:
            dirpath: Path to the dir to save the model.
        """
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        for attr_name in self.attrs_to_save:
            attr_filepath = os.path.join(
                dirpath, '{0}.pickle'.format(attr_name),
            )
            with open(attr_filepath, 'wb') as attr_file:
                attr_value = getattr(self, attr_name)
                pickle.dump(attr_value, attr_file)

    def _preprocess(self, interactions: pd.DataFrame) -> None:
        """Preprocess interactions dataset and set it as an attribute.

        Args:
            interactions: Dataset with interactions.
        """
        self.interactions = interactions
        self._build_watched(self.interactions)
        self._build_mappings(self.interactions)
        self._calculate_item_idf(self.interactions)

    def _filter(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Filter cold users and cold items.

        Args:
            interactions: Dataset with interactions.

        Returns:
            Filtered dataset.
        """
        interactions = interactions.groupby(self.item_column).filter(
            lambda x: len(x) >= self.cold_item_threshold,
        )
        interactions = interactions.groupby(self.user_column).filter(
            lambda x: len(x) >= self.cold_user_threshold,
        )
        return interactions

    def _get_interaction_matrix(
        self, interactions: pd.DataFrame,
    ) -> sp.sparse.coo_matrix:
        """Get interaction matrix from data frame.

        Args:
            interactions: Dataset with interactions.

        Returns:
            Sparse matrix with data from interactions.
        """
        if self.weight_column:
            weights = interactions[self.weight_column].astype(np.float32)
        else:
            weights = np.ones(len(interactions), dtype=np.float32)

        n_items = len(interactions[self.item_column].unique())
        n_users = len(interactions[self.user_column].unique())

        return sp.sparse.coo_matrix(
            (
                weights,
                (
                    interactions[self.item_column].map(
                        self._items_mapping.get,
                    ),
                    interactions[self.user_column].map(
                        self._users_mapping.get,
                    ),
                ),
            ),
            shape=(n_items, n_users),
        )

    def _calculate_item_idf(self, interactions: pd.DataFrame) -> None:
        """Calculate items IDF.

        Args:
            interactions: Dataset with interactions.

        Returns:
            Calculated IDF for each item.
        """
        item_idf = pd.DataFrame.from_dict(
            Counter(interactions[self.item_column].values),
            orient='index',
            columns=['doc_freq'],
        ).reset_index()
        item_idf['idf'] = item_idf['doc_freq'].apply(
            lambda x: self._idf(len(interactions), x),
        )
        self._item_idf = item_idf

    def _idf(self, n: int, x: float) -> float:
        """Calculate IDF.

        Args:
            n: Number of items.
            x: Number of item occurrences.

        Returns:
            IDF value.
        """
        return np.log((1 + n) / (1 + x) + 1)

    def _build_mappings(self, interactions: pd.DataFrame) -> None:
        """Build mappings for IDs.

        Args:
            interactions: Dataset with interactions.
        """
        self._users_inv_mapping = dict(
            enumerate(interactions[self.user_column].unique()),
        )
        self._users_mapping = {
            v: k for k, v in self._users_inv_mapping.items()
        }

        self._items_inv_mapping = dict(
            enumerate(interactions[self.item_column].unique()),
        )
        self._items_mapping = {
            v: k for k, v in self._items_inv_mapping.items()
        }

    def _build_watched(self, interactions: pd.DataFrame) -> None:
        """Build watch lists for each user.

        Args:
            interactions: Dataset with interactions.
        """
        user_groups = interactions.groupby(self.user_column)
        self._watched = user_groups[self.item_column].apply(list)

    def _get_similar_users(
        self, user_id: int, k_users: int,
    ) -> Tuple[List[int], List[float]]:
        """Get similar users with KNN model.

        Args:
            user_id: Query user ID.
            k_users: Number of similar users to get.

        Returns:
            List of similar user IDs and similarity scores.
        """
        user_knn_user_id = self._users_mapping[user_id]
        similar_user_ids, scores = self.user_knn.similar_items(
            user_knn_user_id, N=k_users,
        )
        return (
            list(map(self._users_inv_mapping.get, similar_user_ids)),
            scores,
        )
