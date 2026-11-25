
from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader("document_loader/STATS.pdf")

# Extract pages
documents = loader.load()

print("Number of pages:", len(documents))

# Display extracted text
for doc in documents[:2]:
    print(doc.page_content[:500])
