from dotenv import load_dotenv

from lib.models.embedder import Embedder
from lib.models.builderDB import SimpleChromaBuilder  # путь поправьте под ваш проект

load_dotenv()

embeddings = Embedder().embeddings

builder = SimpleChromaBuilder(embeddings)

vectordb = builder.build_or_load(
    md_folder="Data/raw/articles",
    persist_dir="Data/processed/chromadb",
    collection_name="VectorDB_deepvk_USER-bge-m3",
    chunk_size=800,
    chunk_overlap=120,
)
