import re

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

from config import PDF_PATH, CHROMA_DIR

FIELD_PATTERN = re.compile(
    r"(Movie ID|Movie Name|Genre|Description)\s*:\s*"
    r"(.+?)(?=(?:Movie ID|Movie Name|Genre|Description)\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)

# A 1-2 digit page number fused to the start of a lowercase word at a page
# break, e.g. " 1seeking" -> " seeking". Negative lookahead protects real
# ordinals like 85th / 1st / 2nd / 3rd.
GLUED_PAGE_NUMBER = re.compile(r"\b\d{1,2}(?!st\b|nd\b|rd\b|th\b)(?=[a-z]{2,})")


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        pages.append(text)
    return " ".join(pages)


def _clean(value: str) -> str:
    value = GLUED_PAGE_NUMBER.sub("", value).strip()
    value = re.sub(r"\s+\d{1,3}$", "", value)  # stray trailing page number
    return value.strip()


def parse_movies(full_text: str):
    """Split normalized text into per-movie records, tolerant of field order."""
    blocks = re.split(r"(?=Movie\s+ID\s*:)", full_text, flags=re.IGNORECASE)
    movies = []
    for block in blocks:
        if "movie name" not in block.lower():
            continue
        fields = {
            label.strip().lower(): _clean(value)
            for label, value in FIELD_PATTERN.findall(block)
        }
        if "movie name" in fields:
            movies.append(fields)
    return movies


def build_documents(pdf_path: str = PDF_PATH):
    movies = parse_movies(extract_pdf_text(pdf_path))
    if len(movies) < 90:
        raise RuntimeError(
            f"Only parsed {len(movies)} movie records (expected 100) - the PDF "
            "layout may have changed; inspect FIELD_PATTERN in ingestion.py."
        )
    documents = []
    for m in movies:
        name = m.get("movie name", "Unknown")
        genre = m.get("genre", "Unknown")
        documents.append(
            Document(
                page_content=(
                    f"Movie ID: {m.get('movie id', '')}\nTitle: {name}\n"
                    f"Genre: {genre}\nDescription: {m.get('description', '')}"
                ),
                metadata={
                    "movie_id": m.get("movie id", ""),
                    "movie_name": name,
                    "genre": genre,
                },
            )
        )
    print(f"Parsed {len(documents)} movie records from {pdf_path}.")
    return documents


def main():
    documents = build_documents()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_DIR)
    print(f"Vector store built at ./{CHROMA_DIR} ({len(documents)} documents).")


if __name__ == "__main__":
    main()
