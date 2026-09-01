import sys
from pathlib import Path
from dataclasses import dataclass
from .config import DATA_SOURCE_DIR

@dataclass
class Document:
    path: str
    text: str

supported_formats = {'.txt', '.md'}

def load() -> list[Document]:
    print(f"Loading documents from `{DATA_SOURCE_DIR}`", file=sys.stderr)
    documents = []

    base_dir = Path(DATA_SOURCE_DIR)
    for path in base_dir.rglob('*'):
        if path.suffix in supported_formats:
            documents.append(Document(
                path=str(path),
                text=path.read_text(encoding="utf-8"),
            ))
    return documents