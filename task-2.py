import requests
from bs4 import BeautifulSoup

def scrape_website_content(urls):
    """Scrape and extract textual content from websites."""
    all_text = ""
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            all_text += soup.get_text(separator=" ", strip=True)
        else:
            print(f"Failed to scrape {url}")
    return all_text

def process_and_store_website_embeddings(content):
    """Embed scraped content and store in vector DB."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents([content])

    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

def main_task_2(urls, query):
    # Step 1: Scrape content from websites
    content = scrape_website_content(urls)
    
    # Step 2: Process and store embeddings
    vector_db = process_and_store_website_embeddings(content)
    
    # Step 3: Answer query
    response = answer_query(vector_db, query)
    print("Response:", response)

# Example Usage for Task 2
urls = ["https://www.uchicago.edu/", "https://www.stanford.edu/"]
query = "What programs are offered at these universities?"
main_task_2(urls, query)
