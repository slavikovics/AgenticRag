import os
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to parse text file {file_path}: {e}")
        raise


def parse_pdf_file(file_path: str) -> str:
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(file_path)
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("pypdf not installed. Install with: pip install pypdf")
        raise
    except Exception as e:
        logger.error(f"Failed to parse PDF file {file_path}: {e}")
        raise


def parse_markdown_file(file_path: str) -> str:
    return parse_text_file(file_path)


def parse_csv_file(file_path: str) -> str:
    try:
        import csv
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        text_rows = []
        for row in rows:
            text_rows.append(", ".join(row))
        
        return "\n".join(text_rows)
    except Exception as e:
        logger.error(f"Failed to parse CSV file {file_path}: {e}")
        raise


def parse_json_file(file_path: str) -> str:
    try:
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                text_parts.append(f"{key}: {value}")
            return "\n".join(text_parts)
        elif isinstance(data, list):
            return "\n".join(str(item) for item in data)
        else:
            return str(data)
    except Exception as e:
        logger.error(f"Failed to parse JSON file {file_path}: {e}")
        raise


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not text:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end < len(text):
            for sep in ['. ', '\n', '! ', '? ']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def parse_file(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()
    
    parsers = {
        '.txt': parse_text_file,
        '.text': parse_text_file,
        '.md': parse_markdown_file,
        '.markdown': parse_markdown_file,
        '.pdf': parse_pdf_file,
        '.csv': parse_csv_file,
        '.json': parse_json_file,
    }
    
    parser = parsers.get(ext)
    if not parser:
        logger.warning(f"No specific parser for {ext}, using text parser")
        parser = parse_text_file
    
    return parser(file_path)


def process_file_to_documents(
    file_path: str,
    source_name: str = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    if source_name is None:
        source_name = Path(file_path).name
    
    content = parse_file(file_path)
    chunks = chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap)
    
    documents = []
    for idx, chunk in enumerate(chunks):
        doc = {
            "content": chunk,
            "source": source_name,
            "chunk_id": idx,
            "page": None,
            "metadata": {
                "file_path": str(Path(file_path).absolute()),
                "file_name": Path(file_path).name,
                "file_size": os.path.getsize(file_path),
            },
        }
        documents.append(doc)
    
    logger.info(f"Processed {file_path} into {len(documents)} chunks")
    return documents
