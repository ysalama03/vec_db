from typing import Dict, List, Annotated
import numpy as np
import os
import struct
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        print("OPTIMIZED CODE v9 - IMPROVED 10M ACCURACY")
        self.db_path = database_file_path
        self.index_path = index_file_path
        self.n_clusters = 1000  
        self.n_subvectors = 8 
        self.n_centroids_pq = 256
        self.nprobe = 12
              
        self.centroids = None
        self.pq_codebook = None
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
        else:
            if not os.path.exists(os.path.join(self.index_path, 'centroids.npy')):
                self._build_index()
            else:
                self._load_index()
    
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

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """
        Optimized Retrieval:
        1. Vectorized PQ Distance Calculation
        2. Batched File I/O for Re-ranking
        """
        query = query.flatten()
        
        cluster_distances = np.dot(self.centroids, query)
        closest_clusters = np.argsort(cluster_distances)[-self.nprobe:][::-1]
        
        query_subvectors = []
        subvector_dim = DIMENSION // self.n_subvectors
        for m in range(self.n_subvectors):
            start = m * subvector_dim
            query_subvectors.append(query[start : start+subvector_dim])
            
        pq_dists = np.zeros((self.n_subvectors, self.n_centroids_pq), dtype=np.float32)
        for m in range(self.n_subvectors):
            pq_dists[m] = np.dot(self.pq_codebook[m], query_subvectors[m])

        candidates_score = []
        candidates_id = []
        
        inverted_file = os.path.join(self.index_path, 'inverted_lists.dat')
        
        with open(inverted_file, 'rb') as f:
            for cluster_id in closest_clusters:
                f.seek(cluster_id * 8)
                offset, size = struct.unpack('II', f.read(8))
                
                if size == 0: continue
                
                f.seek(offset)
                data = f.read(size)
                entry_size = 4 + self.n_subvectors
                num_entries = len(data) // entry_size
                
                raw_bytes = np.frombuffer(data, dtype=np.uint8).reshape(num_entries, entry_size)
                ids_bytes = raw_bytes[:, :4].tobytes()
                row_ids = np.frombuffer(ids_bytes, dtype=np.int32)
                codes = raw_bytes[:, 4:]
                
                m_indices = np.arange(self.n_subvectors)
                scores_matrix = pq_dists[m_indices, codes] 
                approx_scores = np.sum(scores_matrix, axis=1)
                
                candidates_score.extend(approx_scores)
                candidates_id.extend(row_ids)

        candidates_score = np.array(candidates_score)
        candidates_id = np.array(candidates_id)
        
        if len(candidates_score) == 0: return []
        
        top_n = min(len(candidates_score), max(top_k * 3, 40))
        top_indices = np.argpartition(candidates_score, -top_n)[-top_n:]
        
        best_ids = candidates_id[top_indices]
        
        del candidates_score
        del candidates_id
        
        total_rows = os.path.getsize(self.db_path) // (DIMENSION * 4)
        mmap_matrix = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(total_rows, DIMENSION))
        
        query_norm = np.linalg.norm(query)
        batch_size = 64
        results = []
        
        for i in range(0, len(best_ids), batch_size):
            batch_ids = best_ids[i:i+batch_size]
            candidate_vectors = np.array(mmap_matrix[batch_ids])
            
            dot_products = np.dot(candidate_vectors, query)
            norms = np.linalg.norm(candidate_vectors, axis=1)
            exact_scores = dot_products / (norms * query_norm + 1e-10)
            results.extend(zip(exact_scores, batch_ids))
            del candidate_vectors
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [rid for _, rid in results[:top_k]]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        """Build IVF+PQ index"""
        
        os.makedirs(self.index_path, exist_ok=True)
        num_records = self._get_num_records()
        all_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        
        # Step 1: Train IVF centroids using MiniBatchKMeans on full dataset
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=DB_SEED_NUMBER,
            batch_size=10000,
            n_init='auto',
        )
        kmeans.fit(all_vectors)
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
        
        # Save centroids
        np.save(os.path.join(self.index_path, 'centroids.npy'), self.centroids)
        
        # Step 2: Train PQ codebook on full dataset
        self.pq_codebook = self._train_pq_codebook(all_vectors)
        
        # Save PQ codebook
        np.save(os.path.join(self.index_path, 'pq_codebook.npy'), self.pq_codebook)
        
        # Step 3: Build inverted lists
        self._build_inverted_lists(num_records)
        
    
    def _train_pq_codebook(self, vectors):
        """Train Product Quantization codebook"""
        subvector_dim = DIMENSION // self.n_subvectors
        codebook = np.zeros((self.n_subvectors, self.n_centroids_pq, subvector_dim), dtype=np.float32)
        
        for m in range(self.n_subvectors):
            start_dim = m * subvector_dim
            end_dim = start_dim + subvector_dim
            subvectors = vectors[:, start_dim:end_dim]
            
            # Use MiniBatchKMeans for each subvector
            kmeans = MiniBatchKMeans(
                n_clusters=self.n_centroids_pq,
                random_state=42 + m,
                batch_size=1000,
                max_iter=50,
                n_init=1
            )
            kmeans.fit(subvectors)
            codebook[m] = kmeans.cluster_centers_
            
        
        return codebook
    
    def _compress_vector(self, vector):
        """Compress a vector using PQ"""
        codes = np.zeros(self.n_subvectors, dtype=np.uint8)
        subvector_dim = DIMENSION // self.n_subvectors
        
        for m in range(self.n_subvectors):
            start_dim = m * subvector_dim
            end_dim = start_dim + subvector_dim
            subvector = vector[start_dim:end_dim]
            
            # Find nearest centroid
            distances = np.linalg.norm(self.pq_codebook[m] - subvector, axis=1)
            codes[m] = np.argmin(distances)
        
        return codes
    
    def _compress_batch(self, vectors):
        """Compress a batch of vectors using PQ (vectorized for speed)"""
        batch_size = len(vectors)
        codes = np.zeros((batch_size, self.n_subvectors), dtype=np.uint8)
        subvector_dim = DIMENSION // self.n_subvectors
        
        for m in range(self.n_subvectors):
            start_dim = m * subvector_dim
            end_dim = start_dim + subvector_dim
            subvectors = vectors[:, start_dim:end_dim]
            
            distances = np.linalg.norm(subvectors[:, None, :] - self.pq_codebook[m][None, :, :], axis=2)
            codes[:, m] = np.argmin(distances, axis=1)
        
        return codes
    
    def _decompress_vector(self, codes):
        """Decompress PQ codes back to approximate vector"""
        vector = np.zeros(DIMENSION, dtype=np.float32)
        subvector_dim = DIMENSION // self.n_subvectors
        
        for m in range(self.n_subvectors):
            start_dim = m * subvector_dim
            end_dim = start_dim + subvector_dim
            vector[start_dim:end_dim] = self.pq_codebook[m, codes[m]]
        
        return vector
    
    def _build_inverted_lists(self, num_records):
        """Build inverted lists for all clusters in a single file"""
        
        # First pass: collect vectors for each cluster
        cluster_data = {cluster_id: [] for cluster_id in range(self.n_clusters)}
        
        # Open database once for bulk reading
        db_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        
        batch_size = 10000
        for batch_start in range(0, num_records, batch_size):
            batch_end = min(batch_start + batch_size, num_records)
            
            # Bulk read batch
            batch_vectors = np.array(db_vectors[batch_start:batch_end])
            
            # Assign to clusters
            distances = np.dot(batch_vectors, self.centroids.T)
            cluster_assignments = np.argmax(distances, axis=1)
            
            # Compress entire batch at once
            batch_codes = self._compress_batch(batch_vectors)
            
            # Store compressed codes
            for i, (cluster_id, codes) in enumerate(zip(cluster_assignments, batch_codes)):
                row_id = batch_start + i
                entry = struct.pack('I', row_id) + codes.tobytes()
                cluster_data[cluster_id].append(entry)
                    
        # Second pass: write all clusters to single file with header
        inverted_file = os.path.join(self.index_path, 'inverted_lists.dat')
        
        with open(inverted_file, 'wb') as f:
            # Reserve space for header (1000 clusters * 8 bytes each)
            header_size = self.n_clusters * 8
            f.write(b'\x00' * header_size)
            
            # Write cluster data and record offsets
            cluster_offsets = []
            for cluster_id in range(self.n_clusters):
                offset = f.tell()
                data = b''.join(cluster_data[cluster_id])
                size = len(data)
                f.write(data)
                cluster_offsets.append((offset, size))
            
            # Write header with offsets and sizes
            f.seek(0)
            for offset, size in cluster_offsets:
                f.write(struct.pack('II', offset, size))
        
    
    def _load_index(self):
        """Load index from disk"""
        # Load centroids
        centroids_path = os.path.join(self.index_path, 'centroids.npy')
        if os.path.exists(centroids_path):
            self.centroids = np.load(centroids_path)
        
        # Load PQ codebook
        pq_path = os.path.join(self.index_path, 'pq_codebook.npy')
        if os.path.exists(pq_path):
            self.pq_codebook = np.load(pq_path)