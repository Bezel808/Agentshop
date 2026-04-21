"""
LlamaIndex-Based Marketplace Provider

使用学术级的 Hybrid Retrieval + Neural Reranking 实现商品搜索。

技术栈:
- BM25 检索（词频匹配）
- Vector Search（语义相似度）
- Reciprocal Rank Fusion（混合策略）
- Cross-encoder Reranking（神经网络重排序）
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import hashlib
import re

# LlamaIndex imports
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Cross-encoder for reranking
from sentence_transformers import CrossEncoder

from aces.environments.protocols import (
    MarketplaceProvider,
    MarketplaceMode,
    Product,
    SearchResult,
    PageState,
)
from aces.config.settings import resolve_datasets_dir
from aces.environments.product_utils import product_from_dict


logger = logging.getLogger(__name__)
INDEX_SCHEMA_VERSION = "v2_unique_product_ids"
QUERY_STOPWORDS = {
    "for", "with", "and", "the", "a", "an", "of", "to", "in",
    "on", "at", "under", "over", "best", "good", "new", "buy",
}
ACCESSORY_HINT_TERMS = {"band", "strap", "chain", "case", "cover", "holder"}


class LlamaIndexMarketplace(MarketplaceProvider):
    """
    学术级的商品搜索引擎，使用 LlamaIndex RAG 技术。
    
    实现了完整的 Retrieve + Rerank pipeline:
    1. BM25 Retrieval (词频检索)
    2. Vector Retrieval (语义检索)  
    3. Hybrid Fusion (RRF 混合)
    4. Neural Reranking (Cross-encoder)
    
    参考文献:
    - BM25: Robertson & Zaragoza (2009)
    - Dense Retrieval: Karpukhin et al. (2020)
    - Reranking: Nogueira & Cho (2019)
    """
    
    def __init__(self):
        """初始化 LlamaIndex marketplace."""
        self.datasets_dir: Optional[Path] = None
        self.index: Optional[VectorStoreIndex] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.vector_retriever: Optional[VectorIndexRetriever] = None
        self.fusion_retriever: Optional[QueryFusionRetriever] = None
        self.reranker: Optional[CrossEncoder] = None
        self.index_cache_dir: Optional[Path] = None
        
        # Product lookup
        self.products: List[Product] = []
        self.product_lookup: Dict[str, Product] = {}
        
        # Current state
        self.current_query: Optional[str] = None
        self.current_results: List[Product] = []
        self.cart: List[Dict] = []
        
        logger.info("Initialized LlamaIndexMarketplace")
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化并构建索引。
        
        Config:
            datasets_dir: ACES 数据集目录
            embedding_model: 向量模型（默认: BAAI/bge-m3）
            reranker_model: 重排序模型（默认: ms-marco-MiniLM-L6-v2）
            use_reranker: 是否使用重排序（默认: True）
            index_cache_dir: 向量索引缓存根目录（默认: .cache/llamaindex）
            rebuild_index: 是否强制重建索引（默认: False）
        """
        self.datasets_dir = resolve_datasets_dir(config.get("datasets_dir"))
        embedding_model = config.get("embedding_model", "BAAI/bge-m3")
        reranker_model = config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L6-v2")
        use_reranker = config.get("use_reranker", True)
        index_cache_dir = config.get("index_cache_dir", ".cache/llamaindex")
        rebuild_index = bool(config.get("rebuild_index", False))
        
        logger.info(f"Building indices from {self.datasets_dir}...")
        
        # 1. 加载所有商品
        self.products = self._load_all_products()
        self.product_lookup = {p.id: p for p in self.products}
        
        logger.info(f"Loaded {len(self.products)} products")
        
        # 2. 转换为 LlamaIndex Documents
        documents = self._products_to_documents()
        
        # 3. 索引缓存路径（按数据集快照 + embedding 模型分桶）
        self.index_cache_dir = self._resolve_index_cache_dir(index_cache_dir)
        cache_key = self._build_index_cache_key(embedding_model)
        persist_dir = self.index_cache_dir / cache_key
        persist_dir.mkdir(parents=True, exist_ok=True)
        vector_store_exists = (
            (persist_dir / "vector_store.json").exists()
            or (persist_dir / "default__vector_store.json").exists()
        )
        can_load_cache = (
            not rebuild_index
            and (persist_dir / "docstore.json").exists()
            and (persist_dir / "index_store.json").exists()
            and vector_store_exists
        )
        
        # 4. 配置 LlamaIndex Settings
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # 5. 创建或加载向量索引
        if can_load_cache:
            logger.info(f"Loading vector index from cache: {persist_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
            self.index = load_index_from_storage(storage_context)
        else:
            logger.info(f"Building vector index (cache miss): {persist_dir}")
            self.index = VectorStoreIndex.from_documents(
                documents,
                show_progress=True,
            )
            self.index.storage_context.persist(persist_dir=str(persist_dir))
            logger.info(f"Persisted vector index to: {persist_dir}")
        
        # 6. 创建 BM25 检索器
        logger.info("Building BM25 retriever...")
        self.bm25_retriever = BM25Retriever.from_defaults(
            index=self.index,
            similarity_top_k=50,  # 召回 50 个
        )
        
        # 7. 创建向量检索器
        logger.info("Building vector retriever...")
        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=50,  # 召回 50 个
        )
        
        # 8. 混合检索使用手动 RRF（避免 LLM 依赖）
        logger.info("Hybrid retrieval ready (manual RRF)...")
        
        # 9. 加载重排序模型
        if use_reranker:
            logger.info(f"Loading reranker: {reranker_model}...")
            self.reranker = CrossEncoder(reranker_model)
        
        logger.info(
            f"✓ LlamaIndex marketplace initialized:\n"
            f"  Products: {len(self.products)}\n"
            f"  Embedding: {embedding_model}\n"
            f"  Index Cache: {persist_dir}\n"
            f"  Reranker: {reranker_model if use_reranker else 'None'}"
        )
    
    def _apply_price_rating_filter(
        self,
        products: List[Product],
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        rating_min: Optional[float] = None,
    ) -> List[Product]:
        """Filter products by price and rating."""
        out = []
        for p in products:
            if price_min is not None and (p.price or 0) < price_min:
                continue
            if price_max is not None and (p.price or 0) > price_max:
                continue
            r = getattr(p, "rating", None) or 0
            if rating_min is not None and r < rating_min:
                continue
            out.append(p)
        return out

    def search_products(
        self,
        query: str,
        sort_by: str = "relevance",
        limit: int = 10,
        page: int = 1,
        price_min: Optional[float] = None,
        price_max: Optional[float] = None,
        rating_min: Optional[float] = None,
        **kwargs
    ) -> SearchResult:
        """
        使用 Hybrid Retrieval + Reranking 搜索商品。支持翻页与价格/星级筛选。
        """
        self.current_query = query
        page = max(1, page)
        page_size = limit
        fetch_limit = max(limit * 10, page * page_size)

        logger.info(f"Searching: '{query}' (limit={fetch_limit})")

        # Stage 1-3: BM25 + Vector + RRF
        bm25_nodes = self.bm25_retriever.retrieve(query)
        vector_nodes = self.vector_retriever.retrieve(query)
        nodes = self._reciprocal_rank_fusion(bm25_nodes, vector_nodes)

        seen_ids = set()
        candidate_products = []
        for node in nodes:
            product_id = node.metadata.get("product_id")
            if product_id and product_id in self.product_lookup and product_id not in seen_ids:
                seen_ids.add(product_id)
                candidate_products.append(self.product_lookup[product_id])

        # Stage 4: Reranking
        if self.reranker and len(candidate_products) > 0:
            pairs = [(query, f"{p.title} {p.description or ''}") for p in candidate_products]
            rerank_scores = self.reranker.predict(pairs)
            sorted_products = [
                p for p, _ in sorted(
                    zip(candidate_products, rerank_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
            ]
        else:
            sorted_products = candidate_products

        # Stage 4.5: lexical relevance re-rank/gate to suppress obvious mismatches
        sorted_products = self._re_rank_by_query_signal(
            products=sorted_products,
            query=query,
            min_keep=max(limit, 8),
        )

        # Apply price/rating filter
        sorted_products = self._apply_price_rating_filter(
            sorted_products, price_min=price_min, price_max=price_max, rating_min=rating_min
        )

        # Apply sort
        if sort_by != "relevance":
            sorted_products = self._sort_products(sorted_products, sort_by)

        total_count = len(sorted_products)
        total_pages = max(1, (total_count + page_size - 1) // page_size) if total_count else 1
        page = min(page, total_pages)
        start = (page - 1) * page_size
        products = sorted_products[start : start + page_size]

        for i, p in enumerate(products):
            p.position = start + i

        self.current_results = products

        logger.info(f"Final results: page {page}/{total_pages}, {len(products)} products")

        return SearchResult(
            query=query,
            products=products,
            total_count=total_count,
            page=page,
            total_pages=total_pages,
            metadata={
                "retrieval": "hybrid",
                "fusion": "reciprocal_rank",
                "reranker": "cross-encoder" if self.reranker else "none",
            }
        )
    
    def get_product_details(self, product_id: str) -> Product:
        """获取商品详情。"""
        if product_id in self.product_lookup:
            return self.product_lookup[product_id]
        
        raise ValueError(f"Product {product_id} not found")
    
    def get_page_state(self) -> PageState:
        """获取当前页面状态。"""
        return PageState(
            products=self.current_results,
            query=self.current_query,
            metadata={
                "mode": "llamaindex",
                "retrieval_tech": "hybrid_rag",
            }
        )
    
    def add_to_cart(self, product_id: str, quantity: int = 1) -> Dict[str, Any]:
        """加入购物车。"""
        product = self.get_product_details(product_id)
        
        self.cart.append({
            "product_id": product_id,
            "product_title": product.title,
            "quantity": quantity,
            "price": product.price,
        })
        
        cart_total = sum(item["price"] * item["quantity"] for item in self.cart)
        
        logger.info(f"Added {quantity}x '{product.title}' to cart")
        
        return {
            "success": True,
            "cart": self.cart,
            "cart_total": cart_total,
        }
    
    def reset(self) -> PageState:
        """重置状态。"""
        self.current_query = None
        self.current_results = []
        self.cart = []
        
        return PageState(products=[], metadata={"mode": "llamaindex"})
    
    def get_mode(self) -> MarketplaceMode:
        """返回模式。"""
        return MarketplaceMode.OFFLINE
    
    def close(self) -> None:
        """清理资源。"""
        logger.info("Closed LlamaIndex marketplace")
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _load_products_data_from_file(
        self, json_file: Path, *, log_skip: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """Load one dataset file and validate it's list[dict]."""
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            if log_skip:
                logger.error(f"Failed to load {json_file}: {e}")
            return None

        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            if log_skip:
                logger.warning(f"Skipping non-product dataset file: {json_file.name}")
            return None
        return data
    
    def _load_all_products(self) -> List[Product]:
        """从所有 JSON 文件加载商品。"""
        all_products = []
        id_counts: Dict[str, int] = {}
        duplicate_count = 0
        
        if not self.datasets_dir.exists():
            logger.warning(f"Datasets directory not found: {self.datasets_dir}")
            return []
        
        # 遍历所有 JSON 文件
        for json_file in sorted(self.datasets_dir.glob("*.json")):
            products_data = self._load_products_data_from_file(json_file, log_skip=True)
            if products_data is None:
                continue

            category = json_file.stem  # 文件名作为类别

            for idx, data in enumerate(products_data):
                data_with_category = {"category": category, **data}
                product = product_from_dict(
                    data_with_category,
                    index=idx,
                    source="llamaindex",
                    category=category,
                )
                base_id = str(product.id)
                seen = id_counts.get(base_id, 0)
                id_counts[base_id] = seen + 1
                if seen > 0:
                    duplicate_count += 1
                    unique_id = f"{base_id}__dup{seen+1}"
                    raw = dict(product.raw_data or {})
                    raw["original_id"] = base_id
                    raw["dedup_id"] = unique_id
                    raw["dedup_rank"] = seen + 1
                    product.id = unique_id
                    product.raw_data = raw
                all_products.append(product)

        if duplicate_count:
            logger.warning(
                "Detected %d duplicate product IDs; auto-renamed to unique IDs during load.",
                duplicate_count,
            )
        return all_products
    
    def _products_to_documents(self) -> List[Document]:
        """转换商品为 LlamaIndex Documents。"""
        documents = []
        
        for product in self.products:
            # 仅使用标题和描述做 embedding（更接近真实电商做法）
            parts = [product.title or ""]
            if product.description:
                parts.append(product.description)
            text = "\n\n".join(p for p in parts if p.strip()).strip() or product.title or "Unknown"
            
            # 创建 Document
            doc = Document(
                text=text,
                metadata={
                    "product_id": product.id,
                    "title": product.title,
                    "price": product.price,
                    "rating": product.rating or 0,
                    "category": product.raw_data.get("category", "unknown"),
                }
            )
            
            documents.append(doc)
        
        return documents
    
    def _reciprocal_rank_fusion(self, nodes_list1, nodes_list2, k=60):
        """
        Reciprocal Rank Fusion 算法。
        
        RRF score = sum(1 / (k + rank_i))
        
        参考: Cormack et al. (2009)
        """
        # 收集所有节点
        all_nodes = {}
        
        # 处理第一个检索器的结果
        for rank, node in enumerate(nodes_list1):
            node_id = node.node_id
            score = 1.0 / (k + rank + 1)
            all_nodes[node_id] = {"node": node, "score": score}
        
        # 处理第二个检索器的结果
        for rank, node in enumerate(nodes_list2):
            node_id = node.node_id
            score = 1.0 / (k + rank + 1)
            if node_id in all_nodes:
                all_nodes[node_id]["score"] += score
            else:
                all_nodes[node_id] = {"node": node, "score": score}
        
        # 按分数排序
        sorted_nodes = sorted(
            all_nodes.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [item["node"] for item in sorted_nodes]
    
    def _sort_products(self, products: List[Product], sort_by: str) -> List[Product]:
        """排序商品。"""
        if sort_by == "price_asc":
            return sorted(products, key=lambda p: p.price)
        elif sort_by == "price_desc":
            return sorted(products, key=lambda p: p.price, reverse=True)
        elif sort_by == "rating":
            return sorted(products, key=lambda p: p.rating or 0, reverse=True)
        else:
            return products

    def _resolve_index_cache_dir(self, raw_dir: str) -> Path:
        """Resolve index cache root path."""
        p = Path(raw_dir)
        if not p.is_absolute():
            # project root: .../ACES-v2
            p = Path(__file__).resolve().parents[2] / p
        return p

    def _build_index_cache_key(self, embedding_model: str) -> str:
        """Build stable cache key from dataset file signatures + embedding model."""
        entries: List[str] = [f"embedding={embedding_model}", f"schema={INDEX_SCHEMA_VERSION}"]
        for fp in sorted(self.datasets_dir.glob("*.json")):
            # Only product datasets should affect index cache invalidation.
            if self._load_products_data_from_file(fp, log_skip=False) is None:
                continue
            try:
                st = fp.stat()
                entries.append(f"{fp.name}:{st.st_size}:{st.st_mtime_ns}")
            except OSError:
                entries.append(f"{fp.name}:missing")
        digest = hashlib.sha1("|".join(entries).encode("utf-8")).hexdigest()[:16]
        return f"idx_{digest}"

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    @staticmethod
    def _compact_text(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    def _extract_query_terms(self, query: str) -> List[str]:
        normalized = self._normalize_text(query)
        terms = [t for t in normalized.split() if len(t) >= 2 and t not in QUERY_STOPWORDS]
        # 保序去重
        return list(dict.fromkeys(terms))

    @staticmethod
    def _term_match(term: str, normalized_text: str, compact_text: str) -> bool:
        # Match both spaced and compact form, e.g. smart watch <-> smartwatch.
        if f" {term} " in f" {normalized_text} ":
            return True
        return term in compact_text

    def _re_rank_by_query_signal(
        self,
        products: List[Product],
        query: str,
        min_keep: int = 10,
    ) -> List[Product]:
        """
        Re-rank by lexical intent and apply light gating for single-word queries.

        Why:
        - Semantic retrievers sometimes return topical but wrong-category items
          (e.g., query "watch" returning "fan").
        - For short queries we prefer title/category hit to keep precision high.
        """
        if not products:
            return products

        terms = self._extract_query_terms(query)
        if not terms:
            return products

        phrase = self._normalize_text(query)
        phrase_compact = self._compact_text(query)
        single_strong_term = len(terms) == 1 and len(terms[0]) >= 4

        scored: List[tuple[float, int, Product, bool]] = []
        for rank, p in enumerate(products):
            title = p.title or ""
            desc = p.description or ""
            category = str((p.raw_data or {}).get("category", ""))

            title_norm = self._normalize_text(title)
            title_compact = self._compact_text(title)
            desc_norm = self._normalize_text(desc)
            desc_compact = self._compact_text(desc)
            cat_norm = self._normalize_text(category)
            cat_compact = self._compact_text(category)

            title_hits = sum(1 for t in terms if self._term_match(t, title_norm, title_compact))
            desc_hits = sum(1 for t in terms if self._term_match(t, desc_norm, desc_compact))
            category_hits = sum(1 for t in terms if self._term_match(t, cat_norm, cat_compact))

            phrase_in_title = bool(phrase and phrase in title_norm) or bool(phrase_compact and phrase_compact in title_compact)
            phrase_in_desc = bool(phrase and phrase in desc_norm) or bool(phrase_compact and phrase_compact in desc_compact)

            core_match = title_hits > 0 or category_hits > 0
            score = 0.0
            score += 100.0 if phrase_in_title else 0.0
            score += 35.0 * title_hits
            score += 12.0 * category_hits
            score += 8.0 * desc_hits
            score += 6.0 if phrase_in_desc else 0.0
            score += 20.0 if core_match else 0.0

            # For single-word intent like "watch", de-prioritize accessory items.
            # Users usually expect the core product first, not strap/chain/case.
            if single_strong_term and title_norm:
                accessory_hit = any(
                    f" {acc} " in f" {title_norm} " for acc in ACCESSORY_HINT_TERMS
                )
                if accessory_hit:
                    score -= 28.0

            score -= rank * 0.03

            scored.append((score, rank, p, core_match))

        # For short single-term queries, suppress non-core matches if we still have enough candidates.
        if single_strong_term:
            core = [row for row in scored if row[3]]
            if len(core) >= min_keep:
                scored = core

        scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        return [p for _, _, p, _ in scored]
