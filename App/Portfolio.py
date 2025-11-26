import pandas as pd
import chromadb
import uuid
import os

class Portfolio:
    def __init__(self, file_path=r"C:\Project\Cold email generator\App\resource\my_portfolio.csv", chroma_dir="vectorstore"):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Portfolio CSV not found at {file_path}")
        self.data = pd.read_csv(file_path)
        # create persistent chroma client; adjust arguments for your chroma version if needed
        self.chroma_client = chromadb.PersistentClient(chroma_dir)
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=[row["Techstack"]],
                                    metadatas=[{"links": row.get("Links", "")}],
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        res = self.collection.query(query_texts=[skills], n_results=2)
        return res.get('metadatas', [])
