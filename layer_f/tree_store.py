from __future__ import annotations

import json
from uuid import uuid5, NAMESPACE_DNS

from layer_f.tree_models import TreeNode

_TREE_CHUNK_TYPE = "page_index_tree"


def _doc_stem_to_point_id(doc_stem: str) -> str:
    chunk_id = f"{doc_stem}__page_index_tree"
    return str(uuid5(NAMESPACE_DNS, chunk_id))


class TreeStore:
    """Manages static (Qdrant) and dynamic (in-memory) PageIndex trees."""

    def __init__(self) -> None:
        # {session_id: {doc_stem: TreeNode}}
        self._dynamic: dict[str, dict[str, TreeNode]] = {}
        # {doc_stem: TreeNode} — cache of loaded static trees
        self._static_cache: dict[str, TreeNode] = {}

    # ── Static trees (Qdrant) ─────────────────────────────────────────────

    def store_static(
        self,
        doc_stem: str,
        tree: TreeNode,
        client,
        collection_name: str,
    ) -> None:
        """Upsert tree as a special Qdrant point (retrieval_weight=0.0)."""
        from qdrant_client.models import PointStruct, SparseVector

        point_id = _doc_stem_to_point_id(doc_stem)
        client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": [0.0] * 1024,
                        "sparse": SparseVector(indices=[], values=[]),
                    },
                    payload={
                        "chunk_type": _TREE_CHUNK_TYPE,
                        "retrieval_weight": 0.0,
                        "doc_stem": doc_stem,
                        "tree_json": json.dumps(tree.to_dict(), ensure_ascii=False),
                    },
                )
            ],
        )
        self._static_cache[doc_stem] = tree

    def load_static(
        self,
        doc_stem: str,
        client,
        collection_name: str,
    ) -> TreeNode | None:
        """Retrieve a previously stored static tree from Qdrant."""
        # Check cache first — client may be None when cache is warm
        if doc_stem in self._static_cache:
            return self._static_cache[doc_stem]
        if client is None:
            return None

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        results, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="chunk_type", match=MatchValue(value=_TREE_CHUNK_TYPE)),
                    FieldCondition(key="doc_stem", match=MatchValue(value=doc_stem)),
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None
        tree_json = results[0].payload.get("tree_json", "")
        try:
            tree = TreeNode.from_dict(json.loads(tree_json))
            # Populate cache on first Qdrant hit
            self._static_cache[doc_stem] = tree
            return tree
        except (json.JSONDecodeError, KeyError):
            return None

    def preload_static(
        self,
        doc_stems: list[str],
        client,
        collection_name: str,
    ) -> list[str]:
        """Eagerly load specified static trees into _static_cache. Returns loaded stems."""
        loaded = []
        for stem in doc_stems:
            tree = self.load_static(stem, client, collection_name)
            if tree is not None:
                loaded.append(stem)
        return loaded

    def list_loaded_stems(self) -> list[str]:
        """Return doc_stems of all currently cached static trees."""
        return list(self._static_cache.keys())

    def get_static_summaries(self) -> dict[str, str]:
        """Return {doc_stem: description} for routing prompt. Fallback: title → stem."""
        return {
            stem: (tree.summary or tree.title or stem)
            for stem, tree in self._static_cache.items()
        }

    # ── Dynamic trees (in-memory) ─────────────────────────────────────────

    def store_dynamic(self, session_id: str, doc_stem: str, tree: TreeNode) -> None:
        """Store a session-scoped tree in memory."""
        self._dynamic.setdefault(session_id, {})[doc_stem] = tree

    def load_dynamic(self, session_id: str, doc_stem: str) -> TreeNode | None:
        """Retrieve a session-scoped tree."""
        return self._dynamic.get(session_id, {}).get(doc_stem)

    def clear_session(self, session_id: str) -> None:
        """Remove all dynamic trees for a session (call when session ends)."""
        self._dynamic.pop(session_id, None)
