import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

class UM6PRetriever:
    """
    Smart retrieval system with filtering and re-ranking
    """

    def __init__(self):
        """Load all necessary components"""
        print("Loading UM6P Retriever...")

        # Load embedding model
        self.embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')

        # Load FAISS index
        self.index = faiss.read_index("um6p_faiss_index.bin")

        # Load metadata
        with open('um6p_chunk_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)

        print(f" Loaded {self.index.ntotal} chunks")
        
        
    def search(self, query, k=5, filters=None):
        """
        Search for relevant chunks

        Args:
            query (str): User question
            k (int): Number of results to return
            filters (dict): Optional filters
                - source_type: 'official_document', 'student_experience', 'community_guide'
                - topics: list of topics to filter by
                - programs: list of programs to filter by
                - countries: list of countries to filter by

        Returns:
            list: Ranked results with metadata
        """
        # IMPROVEMENT: BGE-Small works best with this specific instruction for queries
        instruction = "Represent this sentence for searching relevant passages: "
        query_to_embed = instruction + query

        # Embed the query with the instruction
        query_embedding = self.embedder.encode([query_to_embed], convert_to_numpy=True)

        # Search FAISS (get more than k for filtering)
        search_k = k * 5 if filters else k
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)

        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.metadata[idx]
            similarity = 1 / (1 + dist)

            result = {
                'chunk_id': idx,
                'similarity': similarity,
                'content': chunk['content'],
                'filename': chunk.get('filename', 'Unknown Document'), # Use actual filename
                'source_type': chunk['source_type'],
                'topics': chunk.get('main_topics', []),
                'entities': chunk.get('entities', {}),
                'metadata': chunk.get('metadata', {}),
                'score': float(dist)
            }

            # Add source-specific fields
            if 'program' in chunk:
                result['program'] = chunk['program']
            if 'filename' in chunk:
                result['filename'] = chunk['filename']
            if 'document_type' in chunk:
                result['document_type'] = chunk['document_type']

            results.append(result)

        # Apply filters if provided
        if filters:
            results = self._apply_filters(results, filters)

        # Return top k after filtering
        return results[:k]

    def _apply_filters(self, results, filters):
        """Apply metadata filters to results"""
        filtered = results

        # Filter by source type
        if 'source_type' in filters:
            filtered = [r for r in filtered if r['source_type'] == filters['source_type']]

        # Filter by topics (any match)
        if 'topics' in filters:
            filter_topics = set(filters['topics'])
            filtered = [r for r in filtered if any(t in filter_topics for t in r['topics'])]

        # Filter by programs
        if 'programs' in filters:
            filter_programs = set([p.lower() for p in filters['programs']])
            filtered = [
                r for r in filtered
                if any(p.lower() in filter_programs for p in r['entities'].get('programs', []))
            ]

        # Filter by countries
        if 'countries' in filters:
            filter_countries = set([c.lower() for c in filters['countries']])
            filtered = [
                r for r in filtered
                if any(c.lower() in filter_countries for c in r['entities'].get('countries', []))
            ]

        return filtered

    def get_context_for_llm(self, results):
        """
        Prepare context string for LLM from retrieved chunks

        Args:
            results: List of search results
            max_length: Maximum character length for context

        Returns:
            str: Formatted context with citations
        """
        context_parts = []
        for res in results:
            # Show the document name directly in the context
            header = f"--- Document: {res['filename']} ---"
            context_parts.append(f"{header}\n{res['content']}")
        
        return "\n\n".join(context_parts)