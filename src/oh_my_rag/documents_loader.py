from pathlib import Path
from dataclasses import dataclass
from .config import DATA_SOURE_DIR

@dataclass
class Document:
    path: str
    text: str

supported_formats = {'.txt', '.md'}

def load() -> list[Document]:
    print(f"Loading documents from `{DATA_SOURE_DIR}`")
    documents = []

    base_dir = Path(DATA_SOURE_DIR)
    for path in base_dir.rglob('*'):
        if path.suffix in supported_formats:
            documents.append(Document(
                path=str(path),
                text=path.read_text(encoding="utf-8"),
            ))
    return documents