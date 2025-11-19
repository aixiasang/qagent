import json
import asyncio
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SearchResult:
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class VectorStore(ABC):
    @abstractmethod
    async def add(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        pass

    @abstractmethod
    async def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        pass

    @abstractmethod
    async def delete(self, ids: List[str]):
        pass

    @abstractmethod
    async def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def count(self) -> int:
        pass

    @abstractmethod
    async def save(self):
        pass

    @abstractmethod
    async def load(self):
        pass

    @abstractmethod
    async def clear(self):
        pass


class Shard:
    def __init__(self, shard_id: int, base_path: Path, dimension: Optional[int] = None):
        self.shard_id = shard_id
        self.base_path = base_path
        self.dimension = dimension
        self.ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.count = 0

    def add(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]

        self.ids.extend(ids)

        if self.embeddings is None:
            self.embeddings = embeddings
            if self.dimension is None:
                self.dimension = embeddings.shape[1]
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.count += len(ids)

    def search(self, query_embedding: np.ndarray, k: int) -> List[SearchResult]:
        if self.embeddings is None or self.count == 0:
            return []

        query_norm = np.linalg.norm(query_embedding)
        emb_norms = np.linalg.norm(self.embeddings, axis=1)

        dots = np.dot(self.embeddings, query_embedding)
        scores = dots / (emb_norms * query_norm + 1e-8)

        top_k = min(k, self.count)
        top_k_idx = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_k_idx:
            results.append(
                SearchResult(
                    id=self.ids[idx],
                    text=self.texts[idx],
                    metadata=self.metadatas[idx],
                    score=float(scores[idx]),
                )
            )

        return results

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        results = []
        for target_id in ids:
            if target_id in self.ids:
                idx = self.ids.index(target_id)
                results.append(
                    {
                        "id": self.ids[idx],
                        "text": self.texts[idx],
                        "embedding": self.embeddings[idx].tolist(),
                        "metadata": self.metadatas[idx],
                    }
                )
        return results

    def delete(self, ids: List[str]) -> int:
        deleted_count = 0
        indices_to_delete = []

        for target_id in ids:
            if target_id in self.ids:
                idx = self.ids.index(target_id)
                indices_to_delete.append(idx)
                deleted_count += 1

        if not indices_to_delete:
            return 0

        indices_to_delete.sort(reverse=True)

        for idx in indices_to_delete:
            del self.ids[idx]
            del self.texts[idx]
            del self.metadatas[idx]

        if self.embeddings is not None:
            mask = np.ones(self.count, dtype=bool)
            mask[indices_to_delete] = False
            self.embeddings = self.embeddings[mask]

        self.count -= deleted_count

        return deleted_count

    def save(self):
        self.base_path.mkdir(parents=True, exist_ok=True)

        file_path = self.base_path / f"shard_{self.shard_id}.npz"

        np.savez_compressed(
            file_path,
            ids=np.array(self.ids, dtype=object),
            embeddings=self.embeddings if self.embeddings is not None else np.array([]),
            texts=np.array(self.texts, dtype=object),
            metadatas=np.array(
                [json.dumps(m, ensure_ascii=False) for m in self.metadatas],
                dtype=object,
            ),
            count=np.array([self.count]),
            dimension=np.array([self.dimension if self.dimension else 0]),
        )

    @classmethod
    def load(cls, shard_id: int, base_path: Path):
        file_path = base_path / f"shard_{shard_id}.npz"

        if not file_path.exists():
            raise FileNotFoundError(f"Shard file not found: {file_path}")

        data = np.load(file_path, allow_pickle=True)

        dimension = int(data["dimension"][0]) if data["dimension"][0] > 0 else None
        shard = cls(shard_id, base_path, dimension)

        shard.ids = data["ids"].tolist()
        shard.embeddings = data["embeddings"] if len(data["embeddings"]) > 0 else None
        shard.texts = data["texts"].tolist()
        shard.metadatas = [json.loads(m) for m in data["metadatas"]]
        shard.count = int(data["count"][0])

        return shard


class JsonVectorStore(VectorStore):
    def __init__(self, path: str, shard_size: int = 1000):
        self.path = Path(path)
        self.shard_size = shard_size
        self.index = {
            "version": "1.0",
            "total_count": 0,
            "shards": [],
            "dimension": None,
            "shard_size": shard_size,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self.shards: List[Shard] = []

    async def add(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]

        if self.index["dimension"] is None and len(embeddings) > 0:
            self.index["dimension"] = embeddings.shape[1]

        current_shard = self.shards[-1] if self.shards else None

        if not current_shard or current_shard.count >= self.shard_size:
            shard_id = len(self.shards)
            current_shard = Shard(shard_id, self.path, self.index["dimension"])
            self.shards.append(current_shard)
            self.index["shards"].append(
                {"id": shard_id, "count": 0, "file": f"shard_{shard_id}.npz"}
            )

        current_shard.add(ids, embeddings, texts, metadatas)

        shard_idx = current_shard.shard_id
        self.index["shards"][shard_idx]["count"] = current_shard.count
        self.index["total_count"] += len(ids)
        self.index["updated_at"] = datetime.now().isoformat()

    async def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        all_results = []

        for shard in self.shards:
            results = shard.search(query_embedding, k * 2)
            all_results.extend(results)

        all_results.sort(key=lambda x: x.score, reverse=True)

        return all_results[:k]

    async def delete(self, ids: List[str]):
        total_deleted = 0

        for shard in self.shards:
            deleted = shard.delete(ids)
            total_deleted += deleted

        self.index["total_count"] -= total_deleted
        self.index["updated_at"] = datetime.now().isoformat()

        for i, shard in enumerate(self.shards):
            self.index["shards"][i]["count"] = shard.count

    async def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        results = []

        for shard in self.shards:
            shard_results = shard.get(ids)
            results.extend(shard_results)

        return results

    async def count(self) -> int:
        return self.index["total_count"]

    async def save(self):
        self.path.mkdir(parents=True, exist_ok=True)

        for shard in self.shards:
            shard.save()

        index_path = self.path / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2, ensure_ascii=False)

    async def load(self):
        index_path = self.path / "index.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        self.shard_size = self.index.get("shard_size", 1000)
        self.shards = []

        for shard_info in self.index["shards"]:
            shard = Shard.load(shard_info["id"], self.path)
            self.shards.append(shard)

    async def clear(self):
        for shard in self.shards:
            file_path = self.path / f"shard_{shard.shard_id}.npz"
            if file_path.exists():
                file_path.unlink()

        index_path = self.path / "index.json"
        if index_path.exists():
            index_path.unlink()

        self.shards = []
        self.index = {
            "version": "1.0",
            "total_count": 0,
            "shards": [],
            "dimension": None,
            "shard_size": self.shard_size,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }


class ChromaVectorStore(VectorStore):
    def __init__(self, path: str, collection_name: str = "default"):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        self.path = Path(path)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=str(self.path), settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    async def add(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        if metadatas is None:
            metadatas = [{} for _ in range(len(ids))]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

    async def search(self, query_embedding: np.ndarray, k: int = 5) -> List[SearchResult]:
        results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)

        search_results = []

        if results["ids"] and len(results["ids"]) > 0:
            for i in range(len(results["ids"][0])):
                search_results.append(
                    SearchResult(
                        id=results["ids"][0][i],
                        text=results["documents"][0][i],
                        metadata=(results["metadatas"][0][i] if results["metadatas"] else {}),
                        score=(1.0 - results["distances"][0][i] if results["distances"] else 0.0),
                    )
                )

        return search_results

    async def delete(self, ids: List[str]):
        self.collection.delete(ids=ids)

    async def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        results = self.collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])

        output = []
        for i, id_val in enumerate(results["ids"]):
            output.append(
                {
                    "id": id_val,
                    "text": results["documents"][i] if results["documents"] else "",
                    "embedding": (results["embeddings"][i] if results["embeddings"] else []),
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                }
            )

        return output

    async def count(self) -> int:
        return self.collection.count()

    async def save(self):
        pass

    async def load(self):
        pass

    async def clear(self):
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )


class VectorDB:
    def __init__(
        self,
        embedder,
        store_type: str = "json",
        store_path: str = "./vector_db",
        load_existing: bool = True,
        shard_size: int = 1000,
        collection_name: str = "default",
        **kwargs,
    ):
        self.embedder = embedder
        self.store_path = store_path
        self.store_type = store_type

        if store_type == "json":
            self.store = JsonVectorStore(store_path, shard_size)
        elif store_type == "chroma":
            self.store = ChromaVectorStore(store_path, collection_name)
        else:
            raise ValueError(f"Unknown store_type: {store_type}")

        self._loaded = False

        if load_existing and Path(store_path).exists():
            try:
                asyncio.create_task(self._async_load())
            except:
                pass

    async def _async_load(self):
        if not self._loaded:
            await self.store.load()
            self._loaded = True

    async def add(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embed: bool = True,
    ):
        if embed:
            embed_response = await self.embedder.embed(texts)
            embeddings = np.array([e for e in embed_response.embedding])
        else:
            raise ValueError("Embeddings must be computed. Set embed=True")

        await self.store.add(ids, texts, embeddings, metadatas)

    async def add_with_embeddings(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        await self.store.add(ids, texts, embeddings, metadatas)

    async def search(self, query: str, k: int = 5) -> List[SearchResult]:
        embed_response = await self.embedder.embed([query])
        query_embedding = embed_response.embedding[0]

        results = await self.store.search(query_embedding, k)
        return results

    async def search_with_embedding(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[SearchResult]:
        results = await self.store.search(query_embedding, k)
        return results

    async def delete(self, ids: List[str]):
        await self.store.delete(ids)

    async def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        return await self.store.get(ids)

    async def count(self) -> int:
        return await self.store.count()

    async def save(self):
        await self.store.save()

    async def load(self):
        await self.store.load()
        self._loaded = True

    async def clear(self):
        await self.store.clear()
        self._loaded = False


if __name__ == "__main__":
    from _model import Embedder, get_embedder_cfg

    async def test_vector_db():
        print("=" * 80)
        print("Testing VectorDB with JSON Store")
        print("=" * 80)
        print()

        embedder = Embedder(get_embedder_cfg("ali"))

        vdb = VectorDB(
            embedder=embedder,
            store_type="json",
            store_path="./test_vdb",
            load_existing=False,
            shard_size=3,
        )

        print("1. Adding documents...")
        await vdb.add(
            ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            texts=[
                "Python is a programming language",
                "JavaScript is used for web development",
                "Machine learning is a subset of AI",
                "Deep learning uses neural networks",
                "Natural language processing is part of AI",
            ],
            metadatas=[
                {"category": "programming"},
                {"category": "programming"},
                {"category": "ai"},
                {"category": "ai"},
                {"category": "ai"},
            ],
        )

        count = await vdb.count()
        print(f"✓ Added 5 documents, total count: {count}")
        print()

        print("2. Searching...")
        results = await vdb.search("artificial intelligence", k=3)
        print(f"✓ Search results for 'artificial intelligence':")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.text}")
            print(f"     Score: {result.score:.4f}, ID: {result.id}")
        print()

        print("3. Saving to disk...")
        await vdb.save()
        print(f"✓ Saved to {vdb.store_path}")
        print()

        print("4. Loading from disk...")
        vdb2 = VectorDB(
            embedder=embedder,
            store_type="json",
            store_path="./test_vdb",
            load_existing=True,
        )
        await vdb2.load()

        count2 = await vdb2.count()
        print(f"✓ Loaded database, total count: {count2}")
        print()

        print("5. Searching in loaded database...")
        results2 = await vdb2.search("programming", k=2)
        print(f"✓ Search results for 'programming':")
        for i, result in enumerate(results2, 1):
            print(f"  {i}. {result.text}")
            print(f"     Score: {result.score:.4f}")
        print()

        print("6. Getting specific documents...")
        docs = await vdb2.get(["doc1", "doc3"])
        print(f"✓ Retrieved {len(docs)} documents:")
        for doc in docs:
            print(f"  - {doc['id']}: {doc['text'][:50]}...")
        print()

        print("7. Deleting a document...")
        await vdb2.delete(["doc2"])
        count3 = await vdb2.count()
        print(f"✓ Deleted doc2, new count: {count3}")
        print()

        print("8. Clearing database...")
        await vdb2.clear()
        count4 = await vdb2.count()
        print(f"✓ Cleared database, count: {count4}")
        print()

        print("=" * 80)
        print("All tests completed!")
        print("=" * 80)

    asyncio.run(test_vector_db())
