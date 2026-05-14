"""Tests for StubEmbeddings."""

from __future__ import annotations

import pytest
from src.core.stubs.embedding_stub import StubEmbeddings


class TestStubEmbeddings:
    def test_embed_query_returns_vector_of_correct_dim(self) -> None:
        emb = StubEmbeddings(dimension=768)
        v = emb.embed_query("hello")
        assert len(v) == 768
        # Non-zero so cosine norm is well-defined (pgvector would produce NaN on ||0||).
        assert all(x != 0.0 for x in v)

    def test_query_and_document_vectors_are_identical(self) -> None:
        """Every query and document vector is the same constant, so cosine
        similarity is constant and retrieval falls back to insertion order."""
        emb = StubEmbeddings(dimension=768)
        q = emb.embed_query("anything")
        docs = emb.embed_documents(["one", "two"])
        assert q == docs[0] == docs[1]

    def test_embed_documents_returns_one_vector_per_input(self) -> None:
        emb = StubEmbeddings(dimension=768)
        vs = emb.embed_documents(["a", "b", "c"])
        assert len(vs) == 3
        assert all(len(v) == 768 for v in vs)

    def test_embed_documents_empty_input(self) -> None:
        emb = StubEmbeddings(dimension=768)
        assert emb.embed_documents([]) == []

    def test_custom_dimension_respected(self) -> None:
        emb = StubEmbeddings(dimension=128)
        v = emb.embed_query("x")
        assert len(v) == 128

    @pytest.mark.asyncio
    async def test_aembed_query(self) -> None:
        emb = StubEmbeddings(dimension=768)
        v = await emb.aembed_query("hello")
        assert len(v) == 768

    @pytest.mark.asyncio
    async def test_aembed_documents(self) -> None:
        emb = StubEmbeddings(dimension=768)
        vs = await emb.aembed_documents(["a", "b"])
        assert len(vs) == 2
        assert len(vs[0]) == 768
