import os
import json
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

from dotenv import load_dotenv

"""
Minimal RAG service backed by Pinecone with FinBERT sentence embeddings (dim=768).
Keeps upload, search, chat, and docs listing — trimmed for clarity with no duplicates.
"""

# Embeddings
from sentence_transformers import SentenceTransformer, models

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# File handling
import pypdf
from io import BytesIO

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_FILE = os.path.join(DATA_DIR, "docs.json")


@dataclass
class SearchResult:
    content: str
    metadata: Dict
    score: float


class RAGService:
    def __init__(self):
        print("Initializing RAG Service (Pinecone + Gemini)...")

        # Pinecone setup
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "financial-rag")

    # FinBERT embeddings (ProsusAI/finbert) via SentenceTransformers pooling (dim=768)
    # Note: FinBERT is a finance-tuned BERT-base; sentence embeddings via mean pooling.
        self.embed_dim = 768

        def load_finbert_embedding() -> SentenceTransformer:
            base_model = os.getenv("EMBEDDING_MODEL", "ProsusAI/finbert")
            print(f"• Using FinBERT embeddings ('{base_model}') (dim=768)")
            transformer = models.Transformer(base_model, do_lower_case=True)
            pooling = models.Pooling(
                word_embedding_dimension=transformer.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            return SentenceTransformer(modules=[transformer, pooling])

        self.pc = None
        self.index = None
        self._stats_filter_supported = True
        if not self.pinecone_api_key:
            print("⚠ Warning: PINECONE_API_KEY not set. Add it to .env file")
            # Even without Pinecone, allow local operations to be partially available
            self.embedding_model = load_finbert_embedding()
        else:
            try:
                print(f"Connecting to Pinecone (index: {self.index_name})...")
                self.pc = Pinecone(api_key=self.pinecone_api_key)

                existing = self.pc.list_indexes().names()

                # If index exists, detect and validate dimension (FinBERT requires 768)
                if self.index_name in existing:
                    try:
                        desc = self.pc.describe_index(self.index_name)
                        # desc may be a dict or a model object depending on SDK
                        detected_dim = None
                        if isinstance(desc, dict):
                            detected_dim = (
                                desc.get("dimension")
                                or (desc.get("spec") or {}).get("dimension")
                            )
                        else:
                            detected_dim = getattr(desc, "dimension", None)
                        if detected_dim:
                            self.embed_dim = int(detected_dim)
                            print(f"• Detected existing Pinecone index dimension: {self.embed_dim}")
                            if self.embed_dim != 768:
                                raise ValueError(
                                    f"Pinecone index dim {self.embed_dim} != 768 (FinBERT). "
                                    f"Use an index with dimension=768 or change PINECONE_INDEX_NAME."
                                )
                        else:
                            print("• Could not detect index dimension; using embed_dim=768")
                    except Exception as de:
                        print(f"• Warning: describe_index failed ({de}); proceeding with embed_dim={self.embed_dim}")

                # Load FinBERT embeddings
                self.embedding_model = load_finbert_embedding()

                # Create index if it doesn't exist (with the selected embed_dim)
                if self.index_name not in existing:
                    print(f"Creating Pinecone index: {self.index_name}")
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=768,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region=self.pinecone_environment),
                    )
                self.index = self.pc.Index(self.index_name)
                print("✓ Pinecone connected successfully")
            except Exception as e:
                print(f"✗ Pinecone initialization failed: {e}")
                # Fallback to local embeddings to keep app partially usable
                self.embedding_model = load_finbert_embedding()

        # Chunking params
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 500))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))

    # Document index (local metadata)
        self._docs = self._load_index()

        # LLM (Gemini) - lazy init with fallbacks
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.llm = None
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                # Prefer 2.5 flash, fallback options
                model_candidates = [
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                    "gemini-1.5-flash",
                    "gemini-pro",
                ]
                last_error = None
                for m in model_candidates:
                    try:
                        self.llm = genai.GenerativeModel(m)
                        print(f"✓ Gemini LLM initialized ({m})")
                        last_error = None
                        break
                    except Exception as ie:
                        last_error = ie
                        continue
                if last_error:
                    raise last_error
            except Exception as e:
                print(f"⚠ Gemini LLM initialization failed: {e}")
                self.llm = None

    # ------------------------
    # Local metadata index I/O
    # ------------------------
    def _load_index(self) -> Dict:
        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self):
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, indent=2, ensure_ascii=False)

    # ------------------------
    # Utilities
    # ------------------------
    def _chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        chunks = []
        start = 0
        size = self.chunk_size
        overlap = self.chunk_overlap
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.embedding_model.encode(texts, normalize_embeddings=True)
        try:
            return vecs.tolist()
        except AttributeError:
            return [v for v in vecs]

    # ------------------------
    # Pinecone helpers
    # ------------------------
    def _pinecone_count(self, document_id: str | None = None) -> int:
        """Return total vector count, optionally filtered by a document_id.
        Safe across Pinecone SDK variations and serverless limits."""
        if not self.index:
            return 0
        try:
            # v2 style
            filt = {"document_id": {"$eq": document_id}} if document_id else None
            stats = None
            if filt and self._stats_filter_supported:
                stats = self.index.describe_index_stats(filter=filt)
            else:
                stats = self.index.describe_index_stats()
            # Common shapes: {"namespaces": {"": {"vectorCount": n}}, "totalVectorCount": m}
            if isinstance(stats, dict):
                if "totalVectorCount" in stats:
                    return int(stats.get("totalVectorCount") or 0)
                # Fallback: sum namespaces
                ns = stats.get("namespaces") or {}
                return int(sum((ns[k].get("vectorCount") if isinstance(ns.get(k), dict) else 0) for k in ns))
            else:
                # Model object, try attributes
                tv = getattr(stats, "total_vector_count", None) or getattr(stats, "totalVectorCount", None)
                if tv is not None:
                    return int(tv)
                ns = getattr(stats, "namespaces", None) or {}
                total = 0
                if isinstance(ns, dict):
                    for v in ns.values():
                        vc = getattr(v, "vector_count", None) or getattr(v, "vectorCount", None)
                        if vc is not None:
                            total += int(vc)
                return int(total)
        except Exception as e:
            msg = str(e)
            if "do not support describing index stats with metadata filtering" in msg:
                # Pinecone Serverless/Starter limitation; disable filtered stats to avoid log spam
                self._stats_filter_supported = False
            else:
                print(f"• Warning: describe_index_stats failed: {e}")
            return 0

    # ------------------------
    # Extraction
    # ------------------------
    def _extract_pdf(self, file_bytes: bytes) -> str:
        print(f"\n📄 Extracting PDF ({len(file_bytes)} bytes)...")
        try:
            pdf = pypdf.PdfReader(BytesIO(file_bytes))
            parts = []
            print(f"  PDF has {len(pdf.pages)} pages")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(page_text.strip())
                    print(f"  Page {i+1}: {len(page_text)} chars")
            text = "\n\n".join(parts)
            print(f"✓ Extracted {len(text)} chars total")
            if len(text) < 50:
                raise ValueError("PDF contains insufficient text (may be image-based or encrypted)")
            return text
        except Exception as e:
            print(f"✗ PDF extraction failed: {e}")
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    def _extract_docx(self, file_bytes: bytes) -> str:
        from docx import Document as Docx
        doc = Docx(BytesIO(file_bytes))
        return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    def _extract_csv(self, file_bytes: bytes) -> str:
        """Lightweight CSV to plain text without pandas dependency."""
        import csv
        from io import StringIO
        text = file_bytes.decode("utf-8", errors="ignore")
        reader = csv.reader(StringIO(text))
        rows = [", ".join(row) for row in reader]
        return "\n".join(rows)

    def _extract_text(self, file_bytes: bytes) -> str:
        return file_bytes.decode("utf-8", errors="ignore")

    def _extract_image(self, file_bytes: bytes) -> str:
        raise ValueError("Image OCR not supported. Please use text-based PDFs or documents.")

    def extract_text(self, filename: str, content: bytes) -> str:
        ext = os.path.splitext(filename)[1].lower()
        extractors = {
            ".pdf": self._extract_pdf,
            ".docx": self._extract_docx,
            ".doc": self._extract_docx,
            ".csv": self._extract_csv,
            ".txt": self._extract_text,
        }
        extractor = extractors.get(ext)
        if not extractor:
            raise ValueError(f"Unsupported file type: {ext}")
        try:
            return extractor(content)
        except Exception as e:
            raise ValueError(f"Failed to extract text from {filename}: {str(e)}")

    # ------------------------
    # Core operations
    # ------------------------
    def add_document(self, filename: str, content: bytes) -> Dict:
        print(f"\nProcessing document: {filename} ({len(content)} bytes)")
        if not self.index:
            raise ValueError("Pinecone not configured. Add PINECONE_API_KEY to .env")

        text = self.extract_text(filename, content)
        print(f"Extracted {len(text)} chars")

        chunks = self._chunk_text(text)
        print(f"Created {len(chunks)} chunks")

        doc_id = f"doc_{uuid.uuid4().hex}"

        # Embed and upsert
        vectors = []
        embeddings = self._embed_texts(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            vectors.append({
                "id": f"{doc_id}-{i}",
                "values": vec,
                "metadata": {
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "text": chunk,
                }
            })
        if vectors:
            print(f"Adding {len(vectors)} vectors to Pinecone...")
            self.index.upsert(vectors=vectors)
            print("✓ Added to Pinecone")
            try:
                count = self._pinecone_count(doc_id)
                print(f"• Pinecone now has {count} vectors for document_id={doc_id}")
            except Exception:
                pass

        # Save metadata locally
        self._docs[doc_id] = {
            "document_id": doc_id,
            "filename": filename,
            "file_type": os.path.splitext(filename)[1].replace(".", "").lower(),
            "total_chunks": len(chunks),
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "size_bytes": len(content)
        }
        self._save_index()
        print(f"✓ Document processed: {doc_id}")
        return self._docs[doc_id]

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not self.index:
            return []
        qvec = self._embed_texts([query])[0]
        res = self.index.query(vector=qvec, top_k=top_k, include_metadata=True)
        # Normalize matches from different SDK response shapes
        matches = []
        if isinstance(res, dict):
            matches = res.get("matches") or []
        else:
            matches = getattr(res, "matches", None) or (
                (getattr(res, "data", None) or [{}])[0].get("matches") if getattr(res, "data", None) else []
            )
        out: List[SearchResult] = []
        for m in matches:
            md = m.get("metadata", {})
            content = md.get("text", "")
            raw_score = m.get("score", 0.0)
            try:
                score = float(raw_score)
            except Exception:
                score = 0.0
            if score is None or (isinstance(score, float) and math.isnan(score)):
                score = 0.0
            out.append(SearchResult(content=content, metadata=md, score=score))
        return out

    def answer(self, query: str, top_k: int = 5, include_context: bool = True) -> Tuple[str, List[Dict]]:
        if not self.index:
            return "Pinecone not configured. Add PINECONE_API_KEY to .env", []

        results = self.search(query, top_k)
        context = "\n\n".join([r.content for r in results])
        citations = [
            {
                "filename": r.metadata.get("filename"),
                "document_id": r.metadata.get("document_id"),
                "chunk_index": int(r.metadata.get("chunk_index")) if r.metadata.get("chunk_index") is not None else None,
            }
            for r in results
        ]

        # If no LLM, fall back to a simple metric extractor for common finance asks
        if not self.llm:
            extracted = self._fallback_extract_financials(query, results)
            if extracted:
                return extracted, citations
            answer = f"LLM not configured. Here's the relevant context (top chunks):\n\n{context[:800]}..."
            return answer, citations

        # Prompt: be intelligent, extract figures precisely, cite sources; prefer context but allow light general chat
        base_instructions = (
            "You are a helpful financial analyst assistant.\n"
            "When the user asks for financial figures (e.g., revenue, net income, EBITDA, margins, growth),\n"
            "extract the exact numbers from the provided context. Include: amount, units/currency, period (e.g., FY2024, Q2 2025), and trend if clear.\n"
            "Prefer concise bullets or a small table. Cite sources using filenames and chunk indices provided.\n"
            "Do not invent numbers not present in the context. If unavailable, say so explicitly.\n"
            "You may use general knowledge only for brief definitions or clarification, not for factual financial values.\n"
        )

        if include_context and context.strip():
            prompt = (
                base_instructions
                + "\nContext (verbatim excerpts):\n" + context[:6000] + "\n\n"
                + f"User question: {query}\n"
                + "Answer:"
            )
        else:
            prompt = (
                base_instructions
                + f"\nUser question (no context attached): {query}\n"
                + "If you need specific figures, ask the user to upload or reference documents. Be concise.\n"
                + "Answer:"
            )
        try:
            resp = self.llm.generate_content(prompt)
            text = resp.text if hasattr(resp, "text") else str(resp)
        except Exception as e:
            text = f"Failed to get LLM answer: {e}\n\nContext:\n{context[:800]}..."
        return text, citations

    def _fallback_extract_financials(self, query: str, results: List[SearchResult]) -> str:
        """Very lightweight rule-based extractor for revenue-like asks when LLM is unavailable."""
        q = (query or "").lower()
        want_revenue = any(k in q for k in ["revenue", "net sales", "sales", "turnover"])
        if not want_revenue:
            return ""
        import re
        patterns = [
            r"(?:total\s+)?revenue[s]?\s*(?:\(.*?\))?\s*[:\-]?\s*([$€£₹]?[\d,.]+\s*(?:million|billion|thousand|m|bn|k)?|[\d,.]+)\b",
            r"net\s+sales\s*[:\-]?\s*([$€£₹]?[\d,.]+\s*(?:million|billion|thousand|m|bn|k)?|[\d,.]+)\b",
            r"sales\s*[:\-]?\s*([$€£₹]?[\d,.]+\s*(?:million|billion|thousand|m|bn|k)?|[\d,.]+)\b",
        ]
        hits = []
        for r in results:
            text = r.content or ""
            for pat in patterns:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    val = m.group(1) if m.groups() else m.group(0)
                    if val:
                        hits.append((val.strip(), r))
        if not hits:
            return ""
        # Prepare a concise answer
        bullets = []
        for val, r in hits[:5]:
            src = r.metadata.get("filename") or "document"
            idx = r.metadata.get("chunk_index")
            bullets.append(f"• Revenue: {val} (source: {src}, chunk {idx})")
        return "Here are revenue figures found in your documents:\n" + "\n".join(bullets)

    def list_documents(self) -> List[Dict]:
        # Augment local records with Pinecone counts for better visibility
        docs = []
        for d in self._docs.values():
            doc = dict(d)
            if self.index:
                try:
                    pc_count = self._pinecone_count(doc.get("document_id"))
                    doc["pinecone_vectors"] = pc_count
                    tc = int(doc.get("total_chunks") or 0)
                    doc["in_index"] = pc_count >= tc and tc > 0
                except Exception:
                    pass
            docs.append(doc)
        return docs

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self._docs:
            return False
        try:
            total_chunks = int(self._docs[document_id].get("total_chunks", 0))
            if self.index and total_chunks > 0:
                ids = [f"{document_id}-{i}" for i in range(total_chunks)]
                self.index.delete(ids=ids)
        except Exception as e:
            print(f"Warning: Failed to delete vectors from Pinecone: {e}")
        self._docs.pop(document_id, None)
        self._save_index()
        return True

    def health(self) -> Dict:
        try:
            _ = self._embed_texts(["hello"])
            details = {
                "vector_store": bool(self.index),
                "llm": bool(self.llm),
            }
            # Add Pinecone quick stats if available
            if self.index:
                try:
                    total = self._pinecone_count()
                    details["pinecone_total_vectors"] = total
                    details["embed_dim"] = 768
                    details["index_name"] = self.index_name
                    details["embedding_model"] = os.getenv("EMBEDDING_MODEL", "ProsusAI/finbert")
                except Exception:
                    pass
            return {"status": "healthy", **details}
        except Exception as e:
            return {"status": "degraded", "error": str(e)}
