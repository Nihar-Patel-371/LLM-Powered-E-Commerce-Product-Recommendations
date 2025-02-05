# LLM-Powered-E-Commerce-Product-Recommendations

## Overview  
An end-to-end web mining and LLM based recommendation system for E-commerce, includes hyper-tunning and making it RAG based, for providing better and more accuracte results. The system collects product data from various e-commerce websites, processes it, and uses advanced modeling techniques to recommend products based on user queries.

---

## Key features include:  
- **Web Scraping**: Extracts structured and unstructured product data, leverages APIs, Selenium, and web scraping to extract product details.
- **Data Processing**: Prepares and refines data through feature engineering and cleaning, created labels using Regex from the unstructured fields, to created more accurate results while using them during the training, converts HEX color codes to RGB and clusters products by categories for richer features.
- **LLM**: Used `LangChain` services for orchestrating different tools, hyper-tuned `DistilGPT2` model using `Ollama`.
- **Sentence Transformers**: Embeds textual data into meaningful vector representations using models like `all-mpnet-base-v2` and `all-Mini-L6-v2`.  
- **Cosine Similarity**: Matches user queries with products based on vector similarity for recommendations.  
- **Scalability**: Modular and extensible design for future enhancements, also can be attached with Apache Airflow, and Apache Kafka to automate the extraction and streaming process along with AWS.

---

## Demo

https://github.com/user-attachments/assets/2c74a6d1-1d59-44cd-aa7c-0893b30f29d0


---

## Getting Started  

### Prerequisites  
- Python 3.12+  
- Required libraries: `pandas`, `numpy`, `sentence-transformers`, `scikit-learn`, `webcolors`, `selenium`, etc.  
- Install dependencies using:  
  ```bash
  pip install -r requirements.txt

### Data Scrapping
- Run the `handm.py` file to scrape product details form the website, and to store them locally.
- To run the file:
  ```bash
  py .\handm.py
- The scrapped data will be saved as `handm.pkl`, in the current directory.

### Model Training
- For training the model from the scratch or to update previously existing trained model, by taking the latest scrapped dataset.
- To run the file:
  ```bash
  py .\model_train.py
- The model will be saved as `trained_model.pkl`, in the current directory.

### Model Testing
- To run the flask interface for testing queries and trained model, run the `flask_app.py`.
- To run the file:
  ```bash
  py .\flask_app.py
