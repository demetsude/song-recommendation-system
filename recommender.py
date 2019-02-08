import os
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sps
import sklearn.metrics.pairwise as sk

# File paths for song data, under './data' directory
PATH = {
    'target_playlist': 'target_playlists.csv',
    'target_track': 'target_tracks.csv',
    'playlist_final': 'playlists_final.csv',
    'track_final': 'tracks_final.csv',
    'train': 'train_final.csv'
}


class Recommender:
    def __init__(self):
        '''
            Class definition for the Recommender
        '''
        logging.getLogger().setLevel(logging.INFO)
        self.playlist_id = []
        self.track_id = []
        self.titles_flat = []
        self.owner = []

        self.artist_id = []
        self.albums_flat = []
        self.dic_playlist, self.dic_track = {}, {}

        self.URM = None
        self.rat_user_cbf = None
        self.rat_item_cbf = None
        self.rat_user_cf = None
        self.rat_item_cf = None

        self.target_playlists_df = None
        self.target_tracks_df = None
        # [playlist_id, track_id]
        self.train_final_df = None
        # [created_at, playlist_id, title, numtracks, duration, owner]
        self.playlists_final_df = None
        # [track_id, artist_id, duration, playcount, album, tags]
        self.tracks_final_df = None

    def read_data(self, files=PATH):
        '''
            Reads song data from multiple CSV files
            Input:
                files: dict containing file paths as string
        '''
        self.target_playlists_df = pd.read_csv(
            os.path.join('.', 'data', files['target_playlist']),
            skiprows=[0], header=None)

        self.target_tracks_df = pd.read_csv(os.path.join(
            ".", "data", files['target_track']),
            skiprows=[0], header=None)

        self.train_final_df = pd.read_csv(
            os.path.join(".", "data", files['train']),
            skiprows=[0], header=None, sep='\t')
        self.playlist_id = self.train_final_df.iloc[:, 0].tolist()
        self.track_id = self.train_final_df.iloc[:, 1].tolist()

        self.playlists_final_df = pd.read_csv(
            os.path.join(".", "data", files['playlist_final']),
            skiprows=[0], header=None, sep='\t')
        titles = self.playlists_final_df.iloc[:, 2].tolist()
        titles = [eval(sublist) for sublist in titles]
        self.titles_flat = [item for sublist in titles for item in sublist]
        self.owner = self.playlists_final_df.iloc[:, 5].tolist()

        self.tracks_final_df = pd.read_csv(
            os.path.join(".", "data", files['track_final']),
            skiprows=[0], header=None, sep='\t')
        self.artist_id = self.tracks_final_df.iloc[:, 1].tolist()
        albums_ = self.tracks_final_df.iloc[:, 4].tolist()
        albums_ = [eval(sublist) for sublist in albums_]
        self.albums_flat = [item for sublist in albums_ for item in sublist]

    def _create_URM(self):
        unique_playlist_ids = list(set(self.playlist_id))
        idx = range(0, len(unique_playlist_ids))
        self.dic_playlist = dict(zip(unique_playlist_ids, idx))
        playlist_id_ind = self.train_final_df.iloc[:, 0].\
            apply(lambda x: self.dic_playlist[x]).tolist()

        unique_track_ids = list(set(self.track_id))
        idx = range(0, len(unique_track_ids))
        self.dic_track = dict(zip(unique_track_ids, idx))
        track_id_ind = self.train_final_df.iloc[:, 1].\
            apply(lambda x: self.dic_track[x]).tolist()

        # The number of ratings given in 'train_final.csv' is equal to 1040522.
        rat = np.ones((1040522))
        URM = sps.coo_matrix((rat, (playlist_id_ind, track_id_ind)))
        self.URM = URM.tocsr()

    def _create_ICM(self):
        artist_ids = list(set(self.artist_id))
        idx = range(0, len(artist_ids))
        dic_artist_ids = dict(zip(artist_ids, idx))

        albums = list(set(self.albums_flat))
        idx = range(len(artist_ids), len(artist_ids)+len(albums))
        dic_albums = dict(zip(albums, idx))

        filtered_tracks = self.tracks_final_df[
            self.tracks_final_df.iloc[:, 0].isin(self.track_id)]
        filtered_track_ids = filtered_tracks.iloc[:, 0].tolist()
        filtered_artist_ids = filtered_tracks.iloc[:, 1].tolist()
        filtered_albums = filtered_tracks.iloc[:, 4].tolist()
        filtered_albums = [eval(sublist) for sublist in filtered_albums]
        track_attr_ind, track_ind = [], []
        for idx in range(len(filtered_artist_ids)):
            track_attr_ind.append(dic_artist_ids[filtered_artist_ids[idx]])
            for albm in filtered_albums[idx]:
                track_attr_ind.append(dic_albums[albm])
            len_ind = len(filtered_albums[idx]) + 1
            track_ind.extend(
                [self.dic_track[filtered_track_ids[idx]]] * len_ind)

        content = np.ones((len(track_ind)))
        ICM = sps.coo_matrix((content, (track_ind, track_attr_ind)))
        ICM = ICM.tocsr()
        return ICM

    def _create_UCM(self):
        unique_owners = list(set(self.owner))
        idx = range(0, len(unique_owners))
        dic_owner = dict(zip(unique_owners, idx))

        playlist_attr_ind, playlist_ind = [], []
        filtered_playlists = self.playlists_final_df[
            self.playlists_final_df.iloc[:, 1].isin(self.playlist_id)]
        filtered_playlist_id = filtered_playlists.iloc[:, 1].tolist()
        filtered_owner = filtered_playlists.iloc[:, 5].tolist()
        for idx in range(len(filtered_playlist_id)):
            playlist_attr_ind.append(dic_owner[filtered_owner[idx]])
            playlist_ind.extend([self.dic_playlist[filtered_playlist_id[idx]]])

        content = np.ones((len(playlist_ind)))
        UCM = sps.coo_matrix((content, (playlist_ind, playlist_attr_ind)))
        UCM = UCM.tocsr()
        return UCM

    def filter_seen(self, user_id, sorted_items):
        target_tracks = self.target_tracks_df.iloc[:, 0].tolist()
        target_tracks_new = [self.dic_track[item] for item in target_tracks
                             if item in self.dic_track]
        seen = self.URM[self.dic_playlist[user_id]].indices
        unseen_mask = np.in1d(sorted_items, seen,
                              assume_unique=True, invert=True)
        sorted_items = sorted_items[unseen_mask]
        in_target_tracks_mask = np.in1d(sorted_items, target_tracks_new,
                                        assume_unique=True)
        return sorted_items[in_target_tracks_mask]

    def similarity_matrix_topk(self, sim_matrix, k=50):
        '''
            Extract the most similar <k> entries
            Input:
                sim_matrix: The similarity matrix (NxN)
                in format <csr_matrix>
                k: The number of most similar entries
                to be extracted
            Return:
                w_sparse: Similarity matrix containing only
                k most similar entries in format <csr_matrix>

        '''
        nitems = sim_matrix.shape[0]
        data, rows_ind, cols_indptr = [], [], []

        for item_idx in range(nitems):
            cols_indptr.append(len(data))
            start_pos = sim_matrix.indptr[item_idx]
            end_pos = sim_matrix.indptr[item_idx+1]
            col_data = sim_matrix.data[start_pos:end_pos]
            col_row_index = sim_matrix.indices[start_pos:end_pos]
            idx_sorted = np.argsort(col_data)
            top_k_idx = idx_sorted[-k:]
            data.extend(col_data[top_k_idx])
            rows_ind.extend(col_row_index[top_k_idx])

        cols_indptr.append(len(data))
        w_sparse = sps.csc_matrix((data, rows_ind, cols_indptr),
                                  shape=(nitems, nitems),
                                  dtype=np.float32)
        w_sparse = w_sparse.tocsr()
        return w_sparse

    def estimate_ratings(self):
        '''
            Estimate the ratings using the following four methods:
            - user_cbf: User based content-based filtering
            - item_cbf: Item based content-based filtering
            - usef_cf:  User based collaborative filtering
            - item_cf:  Item based collaborative filtering
        '''
        self._create_URM()
        logging.info("User Rating Matrix (URM) is created.")

        positive_rat = (self.URM > 0).sum(axis=0)
        rat_per_item = np.array(positive_rat).squeeze()
        num_users = self.URM.shape[0]
        weights = np.log10(num_users/rat_per_item)
        normalized_weights = ((weights - np.min(weights)) /
                              np.ptp(weights)).squeeze()
        weights_vector = sps.diags(normalized_weights)
        self.URM = np.dot(self.URM, weights_vector)
        logging.info('URM is multiplied with the item weights')

        UCM = self._create_UCM()
        logging.info('User Content Matrix (UCM) is created.')
        sim_user_cb = sk.cosine_similarity(UCM, dense_output=False)
        sim_user_cb_topk = self.similarity_matrix_topk(sim_user_cb, k=10)
        self.rat_user_cbf = np.dot(sim_user_cb_topk, self.URM)
        logging.info('User based Content based filtering is performed.')

        ICM = self._create_ICM()
        logging.info('Item Content Matrix (ICM) is created.')
        sim_item_cb = sk.cosine_similarity(ICM, dense_output=False)
        sim_item_cb_topk = self.similarity_matrix_topk(sim_item_cb, k=10)
        self.rat_item_cbf = np.dot(self.URM, sim_item_cb_topk)
        logging.info('Item based Content based filtering is performed.')

        sim_item_cf = sk.cosine_similarity(self.URM.T, dense_output=False)
        self.rat_item_cf = np.dot(self.URM, sim_item_cf)
        logging.info('Item based Collaborative filtering is performed.')

        sim_user_cf = sk.cosine_similarity(self.URM, dense_output=False)
        self.rat_user_cf = np.dot(sim_user_cf, self.URM)
        logging.info('User based Collaborative filtering is performed.')

    def recommend(self, user_id, at=5,
                  user_cbf_weight=5, item_cbf_weight=7,
                  user_cf_weight=5, item_cf_weight=6):
        '''
            Recommend songs by averaging the ratings from
            Item CBF, User CBF, Item CF and User CF.
            The weights are decided based on
            numerous observations.
            Input:
                user_id: integer
                at: the number of desired recommendation
                user_cbf_weight: weight for user_cbf results
                item_cbf_weight: weight for item_cbf results
                user_cf_weight: weight for user_cf results
                item_cf_weight: weight for item_cf results
            Return:
                list containing <at> recommendations
        '''
        playlist_id = self.dic_playlist[user_id]
        user_cbf = self.rat_user_cbf[playlist_id, :] * user_cbf_weight
        item_cbf = self.rat_item_cbf[playlist_id, :] * item_cbf_weight

        user_cf = self.rat_user_cf[playlist_id, :] * user_cf_weight
        item_cf = self.rat_item_cf[playlist_id, :] * item_cf_weight

        weight_sum = user_cbf_weight + item_cbf_weight + \
            user_cf_weight + item_cf_weight

        normalized_rat = ((item_cf + user_cf + item_cbf + user_cbf) /
                          weight_sum).toarray()[0]
        sorted_rat = normalized_rat.argsort()[::-1]
        results = self.filter_seen(user_id, sorted_rat)[:at]
        reverse_dic_track = {y: x for x, y in self.dic_track.iteritems()}
        org_results = [reverse_dic_track[x] for x in results]
        org_results = ' '.join(map(str, org_results))
        return org_results

    def recommend_all(self):
        recs_df = self.target_playlists_df
        recs_df.columns = ['playlist_id']
        recs_df['track_ids'] = self.target_playlists_df.iloc[:, 0].\
            apply(lambda x: self.recommend(x))
        recs_df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    rec = Recommender()
    rec.read_data()
    rec.estimate_ratings()
    rec.recommend_all()
