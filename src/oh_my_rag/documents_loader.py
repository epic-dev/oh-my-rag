from pathlib import Path
from .config import DATA_SOURE_DIR

def load_document(path: Path) -> str:
    if path.suffix in {'.txt', '.md'}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"[ERROR]: Unsupported format: {path.suffix}. Supported formats: .txt, .md")

def load():
    documents = []

    base_dir = Path(DATA_SOURE_DIR)
    for path in base_dir.rglob('*'):
        if path.suffix in ['.txt', '.md']:
            print(f"Loading: {path}")
            documents.append({
                "path": str(path),
                "text": load_document(path)
            })
    return documents