import os
import numpy as np
import time
import json
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional

# `faiss` is imported lazily inside the methods that build/save/load the index
# so importing this module (and the scilink.knowledge package) stays cheap.

from ..auth import get_api_key, APIKeyNotFoundError
from ..wrappers.openai_wrapper_embeddings import OpenAIAsEmbeddingModel
from ..wrappers.litellm_wrapper import LiteLLMEmbeddingModel


from ._deprecation import normalize_params

from openai import RateLimitError
from scilink.utils.announce import announce_litellm


def describe_error(e: BaseException) -> str:
    """``type: message`` for a log line — never empty.

    ``str(e)`` is empty for the failures that matter most here: FAISS's
    Python wrapper rejects a query whose dimension differs from the
    index's with a bare ``assert`` (``AssertionError()``), which rendered
    as ``Dense KB retrieval failed ()`` and was undiagnosable (#631).
    """
    return f"{type(e).__name__}: {str(e) or repr(e)}"


class EmbeddingDimensionMismatch(RuntimeError):
    """A query (or new batch) embedding does not fit the dense index: the
    index was built by a different embedding model than the one this
    session embeds with. Named so the failure explains itself and the
    caller's keyword fallback can say why it is running."""


def _index_meta_path(index_path: str) -> Path:
    """Provenance sidecar next to the FAISS index: which embedding model
    built it and its dimension. Shares the index's stem so the planning
    agent's ``default_kb_*`` copies carry it along."""
    return Path(index_path).with_suffix(".meta.json")


class KnowledgeBase:
    """
    Handles embedding, retrieval, and repository structure mapping.
    Supports both Google and OpenAI-compatible (e.g., incubator) embedding models.

    Args:
        api_key: API key for the embedding provider.
        embedding_model: Name of the embedding model.
        base_url: Base URL for internal proxy endpoint.
        use_litellm: If True and base_url is None, use LiteLLM.
        
        google_api_key: DEPRECATED. Use 'api_key' instead.
        local_model: DEPRECATED. Use 'base_url' instead.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        base_url: Optional[str] = None,
        use_litellm: bool = False,
        # Deprecated parameters
        google_api_key: Optional[str] = None,
        local_model: Optional[str] = None,
    ):
   
        # Handle deprecated parameters
        api_key, base_url = normalize_params(
            api_key=api_key,
            google_api_key=google_api_key,
            base_url=base_url,
            local_model=local_model,
            source="KnowledgeBase"
        )
        
        self.embedding_model_name = embedding_model

        # Initialize embedding client. No embedding model is a deliberate
        # choice, not a failure: the KB is KEYWORD-ONLY (BM25 retrieval tier),
        # so no dense embeddings are attempted and no embedding provider /
        # key is needed. Dense retrieval is opt-in — name an embedding model.
        if not embedding_model:
            logging.info("📚 KnowledgeBase: no embedding model — keyword-only "
                         "(BM25) retrieval; dense retrieval disabled.")
            self.embedding_client = None
        elif base_url:
            logging.info(f"🏛️ KnowledgeBase using internal proxy for embeddings")
            self.embedding_client = OpenAIAsEmbeddingModel(
                model=embedding_model,
                api_key=api_key,
                base_url=base_url
            )
        elif use_litellm:
            announce_litellm("KnowledgeBase", embedding_model, embeddings=True)
            self.embedding_client = LiteLLMEmbeddingModel(
                model=embedding_model,
                api_key=api_key
            )
        else:
            logging.info(f"🔷 KnowledgeBase using OpenAI client for embeddings")
            self.embedding_client = OpenAIAsEmbeddingModel(
                model=embedding_model,
                api_key=api_key
            )
            
        self.index = None
        self.chunks = []
        self.sources: List[str | Dict[str, str]] = []
        # Provenance of the loaded/built dense index (None = unrecorded,
        # i.e. files written before the sidecar existed).
        self.index_built_with: Optional[str] = None
        # Why dense retrieval is off for this KB (model or dimension
        # mismatch); ``retrieve`` then serves BM25 without embedding the
        # query — one explanation, not a failed embedding call per query.
        self._dense_disabled_reason: Optional[str] = None
        
        # Registry for Repo Maps: {'repo_name': 'tree_structure_string'}
        # This stores the visual directory trees for any repo you ingest.
        self.repo_maps: Dict[str, str] = {}

    def build(self, chunks: List[Dict[str, any]], batch_size: int = 100):
        """
        Processes a list of text chunks, generates embeddings in batches,
        and builds the vector index.
        """
        import faiss

        if not chunks:
            print("⚠️  KnowledgeBase build skipped: No chunks provided.")
            return

        self.chunks.extend(chunks)

        # No embedding model ⇒ intentional keyword-only KB: keep the chunks,
        # build no dense index, and let the query path's BM25 tier serve
        # retrieval. No embedding call is made (nothing to fail).
        if self.embedding_client is None:
            print("  - 📚 No embedding model configured — building a "
                  "KEYWORD-ONLY knowledge base (BM25 retrieval tier). "
                  "Name an embedding model to enable dense retrieval.")
            self.index = None
            self._bm25_state = None
            return

        texts_to_embed = [chunk['text'] for chunk in chunks]
        all_embeddings = []

        print(f"  - Generating embeddings for {len(texts_to_embed)} chunks using '{self.embedding_model_name}'...")
        
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            batch_num = i // batch_size

            max_retries = 3
            delay = 5 # seconds
            for attempt in range(max_retries):
                try:
                    response = self.embedding_client.embed_content(
                        model=self.embedding_model_name,
                        content=batch_texts,
                        task_type="RETRIEVAL_DOCUMENT" # Ignored by OpenAI wrapper, used by Google
                    )
                    all_embeddings.extend(response['embedding'])
                    print(f"    - Embedded batch {batch_num + 1}/{total_batches}")
                    time.sleep(1) # Small delay to respect API rate limits
                    break # Success
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        print(f"    - ⚠️  Rate limit hit during build. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2 # Exponential backoff
                    else:
                        print(f"    - ❌ Rate limit hit on final attempt. Build failed.")
                        raise e 
                except Exception as e:
                    # Build-time leg of the degradation ladder ("retrieval
                    # is grounding, not a dependency"): an unavailable
                    # embedding provider (e.g. a Bedrock-only session with
                    # the default Gemini embedder) must not abort
                    # generation. Fall back to a KEYWORD-ONLY KB: chunks
                    # are kept, the dense index is dropped entirely (a
                    # partial index would silently search a stale subset),
                    # and the query path's existing BM25 tier takes over.
                    print(f"    - ❌ Error embedding batch {i//batch_size + 1} "
                          f"[{describe_error(e)}]")
                    logging.debug("KB build embedding failure", exc_info=True)
                    print(
                        "  - ⚠️  Embeddings unavailable — building a "
                        "KEYWORD-ONLY knowledge base (BM25 retrieval tier). "
                        "Dense retrieval is disabled for this KB; to enable "
                        "it, configure the embedding provider "
                        f"('{self.embedding_model_name}') and rebuild.")
                    self.index = None
                    self._bm25_state = None
                    return

        embeddings_np = np.array(all_embeddings, dtype=np.float32)
        dimension = embeddings_np.shape[1]

        if self.index is None: 
            print("  - Building FAISS vector index...")
            self.index = faiss.IndexFlatL2(dimension)
        else:
            print("  - Appending to existing FAISS vector index...")
            if dimension != self.index.d:
                # FAISS would reject this with a message-less assert.
                raise EmbeddingDimensionMismatch(self._mismatch_message(
                    dimension, what="new document embeddings"))

        self.index.add(embeddings_np)
        self.index_built_with = self.embedding_model_name
        self._dense_disabled_reason = None
        print(f"  - ✅ Knowledge base built successfully "
              f"({self.index.ntotal} vectors, {dimension}-d, "
              f"'{self.embedding_model_name}').")

    def _mismatch_message(self, got_dim: int, what: str) -> str:
        built = (f"'{self.index_built_with}'" if self.index_built_with
                 else "an unrecorded embedding model")
        return (
            f"{what} are {got_dim}-d but the dense index is "
            f"{self.index.d}-d: the index was built with {built} and this "
            f"session embeds with '{self.embedding_model_name}'. Start the "
            f"session with the model that built the KB, or rebuild the KB "
            f"with '{self.embedding_model_name}'."
        )

    def save(self, index_path: str, chunks_path: str, repo_map_path: str = None, sources_path: str = None):
        """Saves the FAISS index, text chunks, and optionally the repo maps to disk."""
        import faiss

        if self.index:
            faiss.write_index(self.index, index_path)
            print(f"  - FAISS index saved to {index_path}")
            try:
                with open(_index_meta_path(index_path), "w",
                          encoding="utf-8") as f:
                    json.dump({
                        "embedding_model": (self.index_built_with
                                            or self.embedding_model_name),
                        "dimension": int(self.index.d),
                        "vectors": int(self.index.ntotal),
                    }, f, indent=2)
            except Exception as e:  # noqa: BLE001 - provenance is advisory
                print(f"  - ⚠️ Could not write index provenance: {e}")
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2)
            print(f"  - Chunks saved to {chunks_path}")

        with open(sources_path, 'w', encoding='utf-8') as f:
            json.dump(self.sources, f, indent=2)
            print(f"  - Sources saved to {sources_path}")

        # Save Repo Maps Registry
        if repo_map_path and self.repo_maps:
            try:
                with open(repo_map_path, 'w', encoding='utf-8') as f:
                    json.dump(self.repo_maps, f, indent=2)
                print(f"  - Repo maps registry saved to {repo_map_path}")
            except Exception as e:
                print(f"  - ❌ Error saving repo maps: {e}")

    def load(self, index_path: str, chunks_path: str, repo_map_path: str = None, sources_path: str = None) -> bool:
        """Loads a pre-built FAISS index, chunks, and repo maps from disk."""
        import faiss

        index_file = Path(index_path)
        chunks_file = Path(chunks_path)
        sources_file = Path(sources_path)

        if not chunks_file.exists() or not sources_file.exists():
            # Nothing on disk = a fresh KB, not an error; callers narrate the
            # overall outcome (see _load_knowledge_bases / kb_store).
            return False

        try:
            self.index_built_with = None
            self._dense_disabled_reason = None
            if index_file.exists():
                self.index = faiss.read_index(index_path)
                self._read_index_provenance(index_path)
            else:
                # A keyword-only KB (built while embeddings were
                # unavailable) has chunks but no dense index — load it in
                # that mode rather than refusing.
                self.index = None
                print("  - ⚠️  No dense index on disk — loading as a "
                      "KEYWORD-ONLY knowledge base (BM25 retrieval tier).")
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            with open(sources_file, 'r', encoding='utf-8') as f:
                self.sources = json.load(f)
                
            # Load Repo Maps if path provided and file exists
            if repo_map_path and Path(repo_map_path).exists():
                try:
                    with open(repo_map_path, 'r', encoding='utf-8') as f:
                        self.repo_maps = json.load(f)
                    print(f"    - Loaded maps for repos: {list(self.repo_maps.keys())}")
                except Exception as e:
                    print(f"    - ⚠️ Error loading repo maps file: {e}")
            
            n_vec = self.index.ntotal if self.index is not None else 0
            print(f"  - ✅ Successfully loaded {len(self.chunks)} chunks and index with {n_vec} vectors from {len(self.sources)} sources.")
            return True
        except Exception as e:
            print(f"  - ❌ Error loading knowledge base: {e}")
            self.index = None
            self.chunks = []
            return False

    def _read_index_provenance(self, index_path: str) -> None:
        """Pick up the sidecar written by :meth:`save`; on a model mismatch
        say so ONCE and switch this KB to keyword retrieval up front —
        embedding every query only to fail on the index (empty
        ``AssertionError``) was the silent degradation of #631."""
        meta_file = _index_meta_path(index_path)
        if not meta_file.exists():
            return  # pre-sidecar files: the query-time dimension check covers it
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  - ⚠️ Unreadable index provenance {meta_file.name}: {e}")
            return
        self.index_built_with = meta.get("embedding_model") or None
        session_model = self.embedding_model_name
        if (self.index_built_with and session_model
                and self.index_built_with != session_model):
            self._dense_disabled_reason = (
                f"the dense index was built with '{self.index_built_with}' "
                f"but this session embeds with '{session_model}'")
            msg = (f"⚠️  KB dense index was built with "
                   f"'{self.index_built_with}' ({meta.get('dimension', '?')}-d); "
                   f"this session embeds with '{session_model}', so dense "
                   f"retrieval is OFF for this KB and keyword (BM25) search "
                   f"is used. Start the session with embedding model "
                   f"'{self.index_built_with}', or rebuild the KB with "
                   f"'{session_model}'.")
            print(f"  - {msg}")
            logging.warning(msg)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant document chunks for a given query.
        """
        reason = getattr(self, "_dense_disabled_reason", None)
        if self.index and reason:
            print(f"  - ℹ️  Dense retrieval off for this KB ({reason}) — "
                  "retrieving via BM25.")
            return self.retrieve_sparse(query, top_k=top_k)
        if not self.index:
            if self.chunks:
                # Keyword-only KB: the dense tier does not exist, so ANY
                # caller of retrieve() degrades to BM25 here instead of
                # each call site re-implementing the ladder.
                print("  - ℹ️  No dense index (keyword-only KB) — "
                      "retrieving via BM25.")
                return self.retrieve_sparse(query, top_k=top_k)
            print("⚠️  Cannot retrieve: Knowledge base has not been built.")
            return []
            
        print(f"  - Retrieving top {top_k} most relevant chunks for query: '{query[:80]}...'")

        max_retries = 3
        delay = 5 # seconds
        response = None
        for attempt in range(max_retries):
            try:
                response = self.embedding_client.embed_content(
                    model=self.embedding_model_name,
                    content=query,
                    task_type="RETRIEVAL_QUERY" # Ignored by OpenAI wrapper, used by Google
                )
                break # Success
            except RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"    - ⚠️  Rate limit hit embedding query. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2 # Exponential backoff
                else:
                    print(f"    - ❌ Rate limit hit on final attempt. Retrieval failed.")
                    raise e # Re-raise the exception if all retries fail
            except Exception as e:
                print(f"    - ❌ Error embedding query [{describe_error(e)}]")
                logging.debug("KB query embedding failure", exc_info=True)
                raise e
        
        if response is None:
            print("    - ❌ Retrieval failed after retries.")
            return []

        query_embedding = np.array([response['embedding']], dtype=np.float32)

        if query_embedding.ndim == 3:
            query_embedding = np.squeeze(query_embedding, axis=0)

        if query_embedding.shape[1] != self.index.d:
            # FAISS raises a bare AssertionError here — explain it, and
            # stop embedding queries for an index they can never match.
            self._dense_disabled_reason = self._mismatch_message(
                query_embedding.shape[1], what="query embeddings")
            raise EmbeddingDimensionMismatch(self._dense_disabled_reason)

        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve valid chunks (filtering out potential index errors)
        retrieved_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        print(f"  - ✅ Retrieved {len(retrieved_chunks)} chunks.")
        return retrieved_chunks

    def retrieve_sparse(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Keyword (BM25) retrieval over the loaded chunks — needs no
        embedding model, so it works when the provider that built the index
        is unavailable. Same return shape as :meth:`retrieve`. The tokenized
        corpus is cached and invalidated when the chunk count changes."""
        from .sparse_retrieval import bm25_top_k

        cached = getattr(self, "_bm25_state", None)
        hits, state = bm25_top_k(self.chunks, query, top_k=top_k, state=cached)
        self._bm25_state = state
        print(f"  - ✅ Retrieved {len(hits)} chunks via keyword (BM25) search.")
        return hits

    def get_relevant_maps(self, retrieved_chunks: List[Dict]) -> str:
        """
        Dynamic Context Injection:
        Looks at the retrieved chunks, finds which repos they belong to (via 'repo_name' metadata),
        and returns a combined string of ONLY the relevant repo maps.
        """
        relevant_repos = set()
        for chunk in retrieved_chunks:
            # We ensure chunks have this metadata field in planning_agent.py
            repo_name = chunk['metadata'].get('repo_name')
            if repo_name and repo_name in self.repo_maps:
                relevant_repos.add(repo_name)
        
        if not relevant_repos:
            return ""

        combined_map = ""
        for repo in relevant_repos:
            combined_map += f"\n--- DIRECTORY STRUCTURE FOR REPO: {repo} ---\n"
            combined_map += self.repo_maps[repo]
            combined_map += "\n"
            
        return combined_map
       
    def source_difference(self, new_sources: List[str | Dict[str, str]]) -> List[str | Dict[str, str]]:
        """Returns the subset of new sources which are not present in the existing sources."""

        if not new_sources:
            return []

        # Check if the new sources are dictionaries (code repo format with url/ref)
        contains_dict = any(
            isinstance(item, dict) and "path" not in item
            for item in new_sources
        )

        if contains_dict:
            # 1. Convert new sources to tuples for set comparison
            new_sources_tuple = {tuple(sorted(d.items())) for d in new_sources if isinstance(d, dict)}

            # 2. Filter existing sources to ONLY check dictionaries (exclude manifests)
            old_sources_tuple = {
                tuple(sorted(d.items()))
                for d in self.sources
                if isinstance(d, dict) and "path" not in d
            }

            # 3. Calculate difference and convert back to dicts
            difference_tuples = new_sources_tuple - old_sources_tuple
            source_difference = [dict(t) for t in difference_tuples]

        else:
            normalize = lambda p: os.path.normpath(p)

            # Collect existing source info for comparison.
            existing_paths: set[str] = set()
            for s in self.sources:
                if isinstance(s, str):
                    existing_paths.add(normalize(s))
                elif isinstance(s, dict) and "path" in s:
                    existing_paths.add(normalize(s["path"]))

            # Build a set of filenames that were previously ingested.
            # We extract basenames from:
            #   1. Source paths that look like files (have a known extension)
            #   2. Directory manifests stored alongside source paths
            #   3. Chunk metadata (each chunk stores its source file path)
            #   4. Directory sources that still exist on disk
            # This lets us recognise the same file even when it's uploaded
            # to a different session directory whose old path no longer exists.
            _FILE_EXTENSIONS = {
                '.pdf', '.txt', '.md', '.docx', '.csv', '.xlsx', '.tsv',
                '.py', '.json', '.yaml', '.yml', '.npy',
                '.png', '.jpg', '.jpeg', '.tif', '.tiff',
            }
            existing_basenames: set[str] = set()
            for s in self.sources:
                if isinstance(s, dict):
                    # Manifest entry: extract stored filenames
                    if "files" in s:
                        existing_basenames.update(s["files"])
                    continue
                if not isinstance(s, str):
                    continue
                p = Path(normalize(s))
                if p.suffix.lower() in _FILE_EXTENSIONS:
                    # Path string looks like a file — extract basename
                    existing_basenames.add(p.name)
                elif p.is_dir():
                    # Directory still exists on disk — scan it
                    for f in p.rglob("*"):
                        if f.is_file():
                            existing_basenames.add(f.name)
            # Also mine basenames from chunk metadata (most reliable —
            # these are the actual files that were parsed and embedded).
            for chunk in self.chunks:
                meta = chunk.get("metadata", {})
                src = meta.get("source", "")
                if src:
                    existing_basenames.add(Path(src).name)

            source_difference = []
            for s in new_sources:
                ns = normalize(s)
                # 1. Exact path match (fast path)
                if ns in existing_paths:
                    continue
                p = Path(ns)
                # 2. File: check if a file with the same name was already ingested
                if p.is_file():
                    if p.name in existing_basenames:
                        continue
                # 3. Directory: check if ALL files inside were already ingested
                elif p.is_dir():
                    dir_files = {f.name for f in p.rglob("*") if f.is_file()}
                    if dir_files and dir_files.issubset(existing_basenames):
                        continue
                source_difference.append(s)

        # Update history — store directory sources as manifests so that
        # future comparisons work even if the directory is later deleted
        # and contains files (e.g. images) that don't produce chunks.
        for s in source_difference:
            if isinstance(s, str):
                p = Path(normalize(s))
                if p.is_dir():
                    dir_files = sorted(f.name for f in p.rglob("*") if f.is_file())
                    self.sources.append({"path": s, "files": dir_files})
                else:
                    self.sources.append(s)
            else:
                self.sources.append(s)
        return source_difference