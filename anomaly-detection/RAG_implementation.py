# import re

# # Sample contract
# contract = """
# MASTER SERVICE AGREEMENT

# 1. Definitions
# Customer means the organization purchasing the services.
# Service means the software provided by the Company.

# 2. Payment Terms
# Customer shall pay all invoices within 30 days.

# 2.1 Late Payment
# Late payments will incur a penalty of 2% per month.

# 3. Confidentiality
# Both parties agree not to disclose confidential information.

# 4. Termination
# Either party may terminate this agreement with 30 days written notice.
# """

# # Regex to detect numbered headings like:
# # 1. Definitions
# # 2.1 Late Payment
# heading_pattern = re.compile(r'^\d+(\.\d+)?\.\s+.*$', re.MULTILINE)

# # Find all headings
# matches = list(heading_pattern.finditer(contract))

# chunks = []

# for i, match in enumerate(matches):
#     start = match.start()

#     # End is next heading or end of document
#     end = matches[i + 1].start() if i + 1 < len(matches) else len(contract)

#     chunk = contract[start:end].strip()

#     heading = match.group()

#     metadata = {
#         "section": heading,
#         "document": "Master Service Agreement",
#         "version": "v1"
#     }

#     chunks.append({
#         "text": chunk,
#         "metadata": metadata
#     })

# # Display chunks
# for i, chunk in enumerate(chunks):
#     print("=" * 60)
#     print(f"Chunk {i+1}")
#     print("Metadata:", chunk["metadata"])
#     print(chunk["text"])



# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# for chunk in chunks:
#     embedding = model.encode(chunk["text"])

#     # Store in Vector DB
#     vector = {
#         "embedding": embedding,
#         "text": chunk["text"],
#         "metadata": chunk["metadata"]
#     }

#     print(f"Stored: {chunk['metadata']['section']}")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# -------------------------------
# Step 1: Load the PDF
# -------------------------------
loader = PyPDFLoader("contract.pdf")
pages = loader.load()

# -------------------------------
# Step 2: Configure the splitter
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " "
    ]
)

documents = []

# -------------------------------
# Step 3: Chunk the document
# -------------------------------
for page in pages:

    page_number = page.metadata["page"] + 1

    chunks = text_splitter.split_text(page.page_content)

    for i, chunk in enumerate(chunks):

        metadata = {
            "document_name": "contract.pdf",
            "page_number": page_number,
            "chunk_number": i + 1,
            "summary": chunk[:120].replace("\n", " ") + "..."
        }

        # LangChain Document object
        documents.append(
            Document(
                page_content=chunk,
                metadata=metadata
            )
        )

# -------------------------------
# Step 4: Initialize Embedding Model
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# Step 5: Create Vector Embeddings
# -------------------------------
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

print(f"Total Chunks Indexed : {len(documents)}")

# -------------------------------
# Step 6: Query the Vector Store
# -------------------------------
query = "What are the payment terms?"

results = vector_store.similarity_search(
    query=query,
    k=3
)

# -------------------------------
# Step 7: Display Results
# -------------------------------
for result in results:

    print("=" * 70)
    print("Metadata:")
    print(result.metadata)

    print("\nRetrieved Chunk:")
    print(result.page_content)