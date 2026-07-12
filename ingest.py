import re
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

FAQ_FILE = "gigacorp_faq.txt"


def load_faq_documents(file_path):
    docs = []
    current_section = "General"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        # Detect section headers
        if line.startswith("===") and line.endswith("==="):
            current_section = line.replace("===", "").strip()
            continue

        # Detect "Line X:"
        match = re.match(r"^Line\s+(\d+):\s*(.*)$", line)

        if match:
            line_number = int(match.group(1))
            text = match.group(2)
        else:
            line_number = idx
            text = line

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "section": current_section,
                    "line_number": line_number,
                    "source": "gigacorp_faq.txt",
                },
            )
        )

    return docs


# Load documents
documents = load_faq_documents(FAQ_FILE)

# Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create Chroma database
db = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory="./chroma_db",
)

print(f"Stored {len(documents)} documents.")
print("Vector DB created successfully.")