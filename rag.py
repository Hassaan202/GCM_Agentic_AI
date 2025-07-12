import bs4
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
user_agent = "Mozilla/5.0(Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"

# RAG pipeline
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
vector_store = InMemoryVectorStore(embeddings)

web_urls = [
    'https://medlineplus.gov/diabetes.html',
    'https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes',
    'https://www.ncbi.nlm.nih.gov/books/NBK279340/',
    'https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes/continuous-glucose-monitoring',
    'https://www.niddk.nih.gov/health-information/diabetes/overview/healthy-living-with-diabetes',
    'https://www.niddk.nih.gov/health-information/diabetes/overview/insulin-medicines-treatments',
    'https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451',
    'https://www.niddk.nih.gov/health-information/diet-nutrition/changing-habits-better-health'
]

#INDEXING
# Loading the Document
loader = WebBaseLoader(
    web_paths=web_urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=['health-detail-content', 'content', 'bottom']
        )
    ),
    requests_kwargs={
        "headers": {
            "User-Agent": user_agent
        }
    },
    continue_on_failure=True
)
docs = loader.load()

for i, doc in enumerate(docs, 1):
    print(f"=== Document {i} ===")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content length: {len(doc.page_content)} characters")
    print(f"Content preview: {doc.page_content[:500]}...")
    print("-" * 80)
    print()

print(f"Total documents retrieved: {len(docs)}")


# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Storage (Vector Store)
persist_directory = "./vectorstore"
collection_name = "CGM_Agent"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    # Here, we actually create the chroma database using our embedding model
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
except Exception as e:
    print(f"Error setting up ChromaDB: {str(e)}")
    raise

# RETRIEVAL AND GENERATION
# creating a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # K is the amount of chunks to return
)
