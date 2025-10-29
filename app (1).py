import os
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from config import (
    MAX_FILE_SIZE, MAX_FILES_PER_REQUEST, MAX_QUERY_LENGTH,
    SUPPORTED_EXTENSIONS, PORT, DEBUG_MODE
)
from service import RAGService, DATA_DIR

# ============================================================================
# Flask API for Financial RAG Chatbot
# ============================================================================
# Modular backend serving:
# Endpoints:
#  - GET  /api/health
#  - POST /api/upload         (multipart form: files[])
#  - POST /api/chat           { message, topK, includeContext }
#  - GET  /api/docs           list ingested docs
#  - DELETE /api/docs/<id>    delete a document
#  - GET  /                   serve frontend

app = Flask(
    __name__,
    static_folder=os.path.join(os.path.dirname(__file__), "../frontend"),
    static_url_path="/"
)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ============================================================================
# SERVICE INITIALIZATION
# ============================================================================
ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS  # Import from config
rag = RAGService()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file has an allowed extension"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/health")
def health():
    """Health check endpoint - returns service status and configuration"""
    return jsonify(rag.health())


@app.post("/api/upload")
def upload():
    """Upload one or more documents with security validation (single/multi unified)."""
    try:
        print(f"\nUpload request received")

        # Normalize to a list of files (supports both 'file' and 'files')
        files = []
        if "files" in request.files:
            files = request.files.getlist("files")
        elif "file" in request.files:
            files = [request.files["file"]]

        if not files:
            return jsonify({"success": False, "error": "No files uploaded"}), 400
        if len(files) > MAX_FILES_PER_REQUEST:
            return jsonify({"success": False, "error": f"Too many files (max {MAX_FILES_PER_REQUEST})"}), 400

        def handle_file(f):
            if not f or not f.filename:
                return {"success": False, "filename": "unknown", "error": "Invalid file"}
            if not allowed_file(f.filename):
                return {"success": False, "filename": f.filename, "error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}
            content = f.read(MAX_FILE_SIZE + 1)
            if len(content) > MAX_FILE_SIZE:
                return {"success": False, "filename": f.filename, "error": f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)"}
            start = time.perf_counter()
            info = rag.add_document(f.filename, content)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return {
                "success": True,
                "document": {
                    "document_id": info.get("document_id"),
                    "filename": info.get("filename"),
                    "chunksProcessed": info.get("total_chunks") or 0,
                    "processingTime": f"{elapsed_ms} ms",
                }
            }

        results = []
        for f in files:
            try:
                results.append(handle_file(f))
            except ValueError as ve:
                results.append({"success": False, "filename": getattr(f, 'filename', 'unknown'), "error": str(ve)})
            except Exception as e:
                results.append({"success": False, "filename": getattr(f, 'filename', 'unknown'), "error": f"Processing failed: {str(e)}"})

        # For single-file requests, return a simplified shape for frontend compatibility
        if len(files) == 1:
            r0 = results[0]
            status = 200 if r0.get("success") else (400 if "Invalid" in r0.get("error", "") else 500)
            return jsonify(r0), status

        ok = all(r.get("success") for r in results)
        return jsonify({"success": ok, "results": results})

    except Exception as e:
        return jsonify({"success": False, "error": f"Upload failed: {str(e)}"}), 500

@app.post("/api/chat")
def chat():
    """Chat with RAG context with input validation"""
    try:
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        top_k = int(data.get("topK") or 5)
        include_context = bool(data.get("includeContext", True))
        
        # Validation
        if not message:
            return jsonify({"success": False, "error": "message is required"}), 400
        if len(message) > MAX_QUERY_LENGTH:
            return jsonify({"success": False, "error": f"Message too long (max {MAX_QUERY_LENGTH} chars)"}), 400
        if top_k < 1 or top_k > 20:
            return jsonify({"success": False, "error": "topK must be between 1 and 20"}), 400

        answer, citations = rag.answer(message, top_k, include_context)
        return jsonify({
            "success": True,
            "response": answer,
            "citations": citations
        })
    except ValueError as e:
        return jsonify({"success": False, "error": "Invalid input format"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": f"Chat failed: {str(e)}"}), 500


@app.get("/api/docs")
def docs():
    """List all ingested documents"""
    try:
        return jsonify({"success": True, "documents": rag.list_documents()})
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to list documents: {str(e)}"}), 500


@app.delete("/api/docs/<document_id>")
def delete_doc(document_id: str):
    """Delete a document by ID with validation"""
    try:
        # Validate document_id format (should be UUID)
        if not document_id or len(document_id) > 100:
            return jsonify({"success": False, "error": "Invalid document ID"}), 400
        
        ok = rag.delete_document(document_id)
        if not ok:
            return jsonify({"success": False, "error": "Document not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": f"Delete failed: {str(e)}"}), 500


# Optional: serve frontend if you open http://localhost:5000/
@app.get("/")
def index_root():
    frontend_dir = app.static_folder
    return send_from_directory(frontend_dir, "index.html")


if __name__ == "__main__":
    # Configuration imported from config module
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG_MODE)
