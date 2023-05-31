pubmed_processor
================

.. code-block:: python

	
	
	class PubmedAPI:
	    def __init__(self):
	        self.base_url = "https://pubmed.ncbi.nlm.nih.gov/"
	
	    def search(self, query, max_results=10):
	        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
	        record = Entrez.read(handle)
	        handle.close()
	        return record["IdList"]
	
	    def fetch_abstract(self, pubmed_id):
	        handle = Entrez.efetch(
	            db="pubmed", id=pubmed_id, retmode="text", rettype="abstract"
	        )
	        abstract = handle.read()
	        handle.close()
	        return abstract
	
	    def fetch_pmc_full_text(self, pubmed_id):
	        # Get the PMC ID for the PubMed ID
	        handle = Entrez.elink(dbfrom="pubmed", id=pubmed_id, cmd="prlinks")
	        record = Entrez.read(handle)
	        handle.close()
	        pmc_id = None
	        for link in record[0]["LinkSetDb"]:
	            if link["DbTo"] == "pmc":
	                pmc_id = link["Link"][0]["Id"]
	                break
	
	        if not pmc_id:
	            return None
	
	        # Fetch the PMC article XML
	        handle = Entrez.efetch(db="pmc", id=pmc_id, retmode="xml")
	        xml_content = handle.read()
	        handle.close()
	
	        # Parse the XML and extract the full text
	        soup = BeautifulSoup(xml_content, "xml")
	        full_text = " ".join(p.get_text() for p in soup.find_all("p"))
	
	        return full_text
	

.. code-block:: python

	
	
	class PubmedParser:
	    def __init__(self):
	        self.api = PubmedAPI()
	
	    def parse_papers(self, query, max_results=10):
	        pubmed_ids = self.api.search(query, max_results)
	        paper_list = []
	        for pubmed_id in pubmed_ids:
	            paper_dict = {}
	            paper_dict["pubmed_id"] = pubmed_id
	            paper_dict["abstract"] = self.api.fetch_abstract(pubmed_id)
	            paper_dict["content"] = self.api.fetch_pmc_full_text(pubmed_id)
	            paper_list.append(paper_dict)
	        return paper_list
	

.. automodule:: pubmed_processor
   :members:
