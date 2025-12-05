from typing import Dict, List, Annotated
import numpy as np
import os
import pickle

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 64)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k = 5):        
        # Load Index if not loaded
        if not hasattr(self, 'index_data'):
            if os.path.exists(self.index_path):
                with open(self.index_path, 'rb') as f:
                    self.index_data = pickle.load(f)
            else:
                return []

        ivf_centroids = self.index_data['ivf_centroids']
        pq_centroids = self.index_data['pq_centroids']
        inverted_index = self.index_data['inverted_index']
        M = self.index_data['M']
        d_sub = self.index_data['d_sub']
        
        # Normalize Query (for Cosine Similarity via L2)
        norm = np.linalg.norm(query)
        query_norm = query / norm if norm > 0 else query

        # 1. Coarse Search (IVF)
        nprobe = 66
        n_lists = len(ivf_centroids)
        if nprobe > n_lists:
            nprobe = n_lists
            
        # Distance to all coarse centroids
        dists = np.linalg.norm(query_norm - ivf_centroids, axis=1)
        nearest_clusters = np.argsort(dists)[:nprobe]
        
        # 2. Fine Search (PQ with ADC)
        candidate_scores = []
        
        for cluster_id in nearest_clusters:
            if cluster_id not in inverted_index:
                continue
                
            cluster_data = inverted_index[cluster_id]
            ids = cluster_data['ids']
            codes = cluster_data['codes'] # (N_c, M)
            
            if len(ids) == 0:
                continue

            # Residual query for this cluster: q - C_i
            residual_query = query_norm - ivf_centroids[cluster_id] # (1, 64)
            
            # Compute distance table for this cluster (M, 256)
            # This table stores d(sub_q, sub_c)^2
            dist_table = np.zeros((M, 256), dtype=np.float32)
            for m in range(M):
                sub_query = residual_query[0, m*d_sub : (m+1)*d_sub]
                sub_centroids = pq_centroids[m] # (256, d_sub)
                # Squared Euclidean distance
                d = np.linalg.norm(sub_centroids - sub_query, axis=1) ** 2
                dist_table[m] = d
            
            # Look up distances for all vectors in this cluster
            # Sum distances across M subspaces
            dists = np.zeros(len(ids), dtype=np.float32)
            for m in range(M):
                dists += dist_table[m, codes[:, m]]
            
            for i in range(len(ids)):
                candidate_scores.append((dists[i], ids[i]))
                
        # Sort by distance (lowest L2 distance = highest Cosine Similarity)
        candidate_scores.sort(key=lambda x: x[0])
        
        return [s[1] for s in candidate_scores[:top_k]]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        # 1. Load Data
        records = self.get_all_rows()
        n_records, dim = records.shape
        
        # Normalize vectors for Cosine Similarity (L2 on normalized vectors)
        norms = np.linalg.norm(records, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_records = records / norms

        # 2. IVF Training (Coarse Quantizer)
        # Heuristic: sqrt(N) clusters
        n_lists = int(np.sqrt(n_records))
        if n_lists < 1: n_lists = 1
        
        rng = np.random.default_rng(DB_SEED_NUMBER)
        
        # Initialize IVF centroids
        if n_records >= n_lists:
            indices = rng.choice(n_records, n_lists, replace=False)
            ivf_centroids = normalized_records[indices].copy()
        else:
            ivf_centroids = normalized_records.copy()
            n_lists = n_records

        # Simple K-Means for IVF
        for _ in range(15): 
            # Assign
            labels = np.zeros(n_records, dtype=int)
            chunk_size = 1024
            for i in range(0, n_records, chunk_size):
                chunk = normalized_records[i:i+chunk_size]
                dists = np.linalg.norm(chunk[:, None] - ivf_centroids[None, :], axis=2)
                labels[i:i+chunk_size] = np.argmin(dists, axis=1)
            
            # Update
            new_centroids = np.zeros_like(ivf_centroids)
            counts = np.zeros(n_lists)
            for i in range(n_records):
                l = labels[i]
                new_centroids[l] += normalized_records[i]
                counts[l] += 1
            
            for k in range(n_lists):
                if counts[k] > 0:
                    new_centroids[k] /= counts[k]
                else:
                    new_centroids[k] = ivf_centroids[k]
            
            if np.allclose(ivf_centroids, new_centroids, atol=1e-4):
                break
            ivf_centroids = new_centroids

        # Final Assignment to Clusters
        ivf_labels = np.zeros(n_records, dtype=int)
        for i in range(0, n_records, 1024):
            chunk = normalized_records[i:i+1024]
            dists = np.linalg.norm(chunk[:, None] - ivf_centroids[None, :], axis=2)
            ivf_labels[i:i+1024] = np.argmin(dists, axis=1)

        # 3. PQ Training (Product Quantization)
        M = 8 # Number of subspaces
        d_sub = dim // M
        Ks = 256 # Centroids per subspace
        
        # Calculate residuals (vector - coarse_centroid)
        residuals = normalized_records - ivf_centroids[ivf_labels]
        
        pq_centroids = np.zeros((M, Ks, d_sub), dtype=np.float32)
        
        for m in range(M):
            sub_data = residuals[:, m*d_sub : (m+1)*d_sub]
            
            # Init PQ centroids
            if n_records >= Ks:
                indices = rng.choice(n_records, Ks, replace=False)
                centroids = sub_data[indices].copy()
            else:
                indices = rng.choice(n_records, Ks, replace=True)
                centroids = sub_data[indices].copy()

            # Train PQ K-Means
            for _ in range(15):
                # Assign
                sub_labels = np.zeros(n_records, dtype=int)
                for i in range(0, n_records, 1024):
                    chunk = sub_data[i:i+1024]
                    d = np.linalg.norm(chunk[:, None] - centroids[None, :], axis=2)
                    sub_labels[i:i+1024] = np.argmin(d, axis=1)
                
                # Update
                new_centroids = np.zeros_like(centroids)
                counts = np.zeros(Ks)
                for i in range(n_records):
                    l = sub_labels[i]
                    new_centroids[l] += sub_data[i]
                    counts[l] += 1
                
                for k in range(Ks):
                    if counts[k] > 0:
                        new_centroids[k] /= counts[k]
                    else:
                        new_centroids[k] = centroids[k]
                
                if np.allclose(centroids, new_centroids, atol=1e-4):
                    break
                centroids = new_centroids
            
            pq_centroids[m] = centroids

        # 4. Encode Database
        codes = np.zeros((n_records, M), dtype=np.uint8)
        for m in range(M):
            sub_data = residuals[:, m*d_sub : (m+1)*d_sub]
            centroids = pq_centroids[m]
            for i in range(0, n_records, 1024):
                chunk = sub_data[i:i+1024]
                d = np.linalg.norm(chunk[:, None] - centroids[None, :], axis=2)
                codes[i:i+1024, m] = np.argmin(d, axis=1)

        # 5. Build Inverted Index
        inverted_index = {}
        for i in range(n_records):
            cluster = ivf_labels[i]
            if cluster not in inverted_index:
                inverted_index[cluster] = {'ids': [], 'codes': []}
            inverted_index[cluster]['ids'].append(i)
            inverted_index[cluster]['codes'].append(codes[i])
            
        # Convert lists to arrays for efficiency
        for cluster in inverted_index:
            inverted_index[cluster]['ids'] = np.array(inverted_index[cluster]['ids'], dtype=np.int32)
            inverted_index[cluster]['codes'] = np.array(inverted_index[cluster]['codes'], dtype=np.uint8)

        # Save Index
        index_data = {
            'ivf_centroids': ivf_centroids,
            'pq_centroids': pq_centroids,
            'inverted_index': inverted_index,
            'M': M,
            'd_sub': d_sub
        }
        
        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f)


