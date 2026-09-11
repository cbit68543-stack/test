#!/usr/bin/env python3
"""
Private RAG - Simplified Version with Qwen2.5
- Uses keyword search instead of embeddings (much faster)
- No external APIs needed (fully private, runs on Ollama)
- Works with PDF and DOCX files immediately
"""

import sys
import requests
import re
from pathlib import Path
from typing import List, Dict
from PyPDF2 import PdfReader
from docx import Document as DocxDocument


# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5"
DATA_FOLDER = "data"

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    text = []
    try:
        reader = PdfReader(file_path)
        print(f"   📄 Pages found: {len(reader.pages)}")
        for page_num, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted.strip():
                text.append(extracted)
            else:
                print(f"   ⚠️  Page {page_num + 1}: No text extracted")
    except Exception as e:
        print(f"❌ Error reading PDF {file_path}: {e}")
        return ""

    result = "\n".join(text)
    if not result.strip():
        print(f"   ⚠️  WARNING: PDF appears to be empty or text couldn't be extracted")
    return result

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    text = []
    try:
        doc = DocxDocument(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text.append(para.text)
    except Exception as e:
        print(f"❌ Error reading DOCX {file_path}: {e}")
    return "\n".join(text)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    words = text.split()
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        chunk_text = " ".join(current_chunk)

        if len(chunk_text) >= chunk_size:
            chunks.append(chunk_text)
            # Keep last overlap words for next chunk
            current_chunk = current_chunk[-(overlap // 5):]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def keyword_search(chunks: List[str], question: str, k: int = 3) -> List[str]:
    """Find relevant chunks using keyword matching"""
    # Extract keywords from question
    keywords = re.findall(r'\b\w{4,}\b', question.lower())

    scores = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Score based on keyword matches
        score = sum(chunk_lower.count(kw) for kw in keywords)
        scores.append((chunk, score))

    # Return top k chunks by score
    scores.sort(key=lambda x: x[1], reverse=True)

    # If no keyword matches found, return first k chunks
    if scores[0][1] == 0:
        return chunks[:k]

    return [chunk for chunk, _ in scores[:k]]

def load_documents() -> Dict[str, str]:
    """Load PDF and DOCX files from the data folder"""
    documents = {}

    if not Path(DATA_FOLDER).exists():
        print(f"Creating '{DATA_FOLDER}' folder...")
        Path(DATA_FOLDER).mkdir(exist_ok=True)
        print(f"✅ Please add your PDF and DOCX files to the '{DATA_FOLDER}' folder.")
        return documents

    # Load PDFs
    for pdf_file in Path(DATA_FOLDER).glob("*.pdf"):
        print(f"📄 Loading PDF: {pdf_file.name}")
        text = extract_text_from_pdf(str(pdf_file))
        if text.strip():
            documents[pdf_file.name] = text
            print(f"   ✅ Loaded {len(text)} characters")
        else:
            print(f"   ⚠️  SKIPPED: PDF is empty or couldn't extract text")

    # Load DOCX files
    for docx_file in Path(DATA_FOLDER).glob("*.docx"):
        print(f"📄 Loading DOCX: {docx_file.name}")
        text = extract_text_from_docx(str(docx_file))
        if text.strip():
            documents[docx_file.name] = text
            print(f"   ✅ Loaded {len(text)} characters")
        else:
            print(f"   ⚠️  SKIPPED: DOCX is empty")

    if documents:
        print(f"✅ Loaded {len(documents)} documents")
    else:
        print(f"⚠️  No readable documents found!")
    return documents

def build_chunk_db(documents: Dict[str, str]) -> Dict:
    """Build chunk database (fast, no embeddings)"""
    print("\n📚 Processing documents into chunks...")

    chunk_db = {
        "chunks": [],
        "metadata": [],
        "doc_names": list(documents.keys())
    }

    total_docs = len(documents)
    total_chunks = 0

    for idx, (doc_name, content) in enumerate(documents.items(), 1):
        chunks = chunk_text(content)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_db["chunks"].append(chunk)
            chunk_db["metadata"].append({
                "doc": doc_name,
                "chunk": chunk_idx
            })
            total_chunks += 1

        print(f"  [{idx}/{total_docs}] {doc_name}: {len(chunks)} chunks ✅")

    print(f"\n✅ Total chunks created: {total_chunks}")
    return chunk_db

def query_llm_with_context(context: str, question: str) -> str:
    """Query LLM with context"""
    prompt = f"""Based on the following context from documents, answer the question clearly and concisely.
If the answer is not found in the context, say "I don't have that information in the documents."

Context:
{context}

Question: {question}

Answer:"""

    try:
        print(f"   🤖 Querying Qwen2.5...")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "No response from model").strip()
            return answer
        else:
            return f"❌ Error from Ollama: {response.status_code}"
    except Exception as e:
        return f"❌ Error querying LLM: {e}"

def ask_question(chunk_db: Dict, question: str) -> str:
    """Ask question using keyword search and Qwen2.5"""
    if not chunk_db["chunks"]:
        return "❌ No documents indexed. Please add PDF or DOCX files to the 'data' folder."

    print(f"\n🔍 Searching for relevant content...")

    # Keyword search for relevant chunks
    relevant_chunks = keyword_search(chunk_db["chunks"], question, k=5)


    # Build context
    context = "\n\n---\n\n".join(relevant_chunks)

    # Get answer from Qwen2.5
    answer = query_llm_with_context(context, question)

    return answer

def main():
    """Main application"""
    print("\n" + "=" * 70)
    print(" PRIVATE RAG with Qwen2.5 (Ollama)")
    print(" Fully Local - No External APIs")
    print("=" * 70)

    # Check Ollama connection
    print("\n🔗 Checking Ollama connection...")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"   Available models: {', '.join(model_names[:5])}")
        else:
            print(f"❌ Ollama is not responding properly")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print(f"   Make sure Ollama is running at {OLLAMA_BASE_URL}")
        sys.exit(1)

    # Load documents
    print("\n📚 Loading documents...")
    documents = load_documents()

    if not documents:
        print("\n⚠️ No documents loaded yet!")
        print(f"   Please add PDF or DOCX files to the '{DATA_FOLDER}' folder and run again.")
        return

    # Build chunk database
    chunk_db = build_chunk_db(documents)

    # Interactive Q&A loop
    print("\n" + "=" * 70)
    print(" Ready to answer questions!")
    print(" Type 'quit', 'exit' or 'q' to exit")
    print("=" * 70)

    while True:
        question = input("\n📝 Ask a question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(chunk_db, question)
        print(f"\n💡 Answer:\n{answer}")

if __name__ == "__main__":
    main()
