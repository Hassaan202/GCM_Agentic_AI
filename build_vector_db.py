import os
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Load API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
user_agent = "Mozilla/5.0(Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"


# URLs to scrape
web_urls = [
    'https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes',
    'https://www.ncbi.nlm.nih.gov/books/NBK279340/',
    'https://www.niddk.nih.gov/health-information/diabetes/overview/healthy-living-with-diabetes',
    'https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451',
    'https://www.niddk.nih.gov/health-information/diet-nutrition/changing-habits-better-health'
]


# Load and parse HTML
loader = WebBaseLoader(
    web_paths=web_urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=['health-detail-content', 'bottom', 'main-content', 'content']
        )
    ),
    requests_kwargs={"headers": {"User-Agent": user_agent}},
    continue_on_failure=True
)

docs = loader.load()


# Filter noisy/irrelevant docs
def filter_documents(docs):
    filtered = []
    for doc in docs:
        content = doc.page_content.lower()
        if (
            "subscription" not in content and
            "sorry something went wrong" not in content and
            len(doc.page_content.strip()) > 200 and
            "retry" not in content
        ):
            filtered.append(doc)
    return filtered

docs = filter_documents(docs)


with open("file.txt", "w") as f:
    for i, doc in enumerate(docs, 1):
        f.write(f"=== Document {i} ===\n")
        f.write(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        f.write(f"Content length: {len(doc.page_content)} characters\n")
        f.write(f"Content preview: {doc.page_content}\n")


# Text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
splits = splitter.split_documents(docs)


# Store in persistent Chroma DB
persist_dir = "./vectorstore"
collection_name = "CGM_Agent"


if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)


try:
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )
    print("Index successfully built and stored.")
except Exception as e:
    print(f"Error during indexing: {str(e)}")
