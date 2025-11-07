import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text_file(file_path, chunk_size=1000, chunk_overlap=200):
    """
    Read a cleaned text file and split it into overlapping chunks.
    Returns a list of chunks (each as a dict with id + text).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    docs = splitter.split_text(text)

    chunks = [{"chunk_id": i, "text": doc} for i, doc in enumerate(docs)]
    return chunks


def process_clean_dir(input_dir, output_dir="data/chunks", chunk_size=1000, chunk_overlap=200):
    """
    Iterates over all cleaned text files, chunks them, and stores JSON outputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            file_path = os.path.join(root, file)
            chunks = chunk_text_file(file_path, chunk_size, chunk_overlap)

            # Output path mirrors folder structure
            rel_dir = os.path.relpath(root, input_dir)
            save_dir = os.path.join(output_dir, rel_dir)
            os.makedirs(save_dir, exist_ok=True)

            base_name = file.replace(".txt", "_chunks.json")
            save_path = os.path.join(save_dir, base_name)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

            print(f"📄 Chunked {file} → {len(chunks)} chunks saved at {save_path}")


if __name__ == "__main__":
    print("🔪 Starting text chunking ...")
    process_clean_dir("data/clean/sec")
    process_clean_dir("data/clean/news")
    process_clean_dir("data/clean/market")
    print("🎯 All cleaned data successfully chunked → data/chunks/")
