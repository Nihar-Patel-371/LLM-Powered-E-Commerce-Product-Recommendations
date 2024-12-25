# LLM-Powered-E-Commerce-Product-Recommendations

## Overview  
An end-to-end web mining and LLM based recommendation system for E-commerce. The system collects product data from various e-commerce websites, processes it, and uses advanced modeling techniques to recommend products based on user queries.

---

## Key features include:  
- **Web Scraping**: Extracts structured and unstructured product data, leverages APIs, Selenium, and web scraping to extract product details.
- **Data Processing**: Prepares and refines data through feature engineering and cleaning, created labels using Regex from the unstructured fields, to created more accurate results while using them during the training, converts HEX color codes to RGB and clusters products by categories for richer features.
- **Sentence Transformers**: Embeds textual data into meaningful vector representations using models like `all-mpnet-base-v2` and `all-Mini-L6-v2`.  
- **Cosine Similarity**: Matches user queries with products based on vector similarity for recommendations.  
- **Scalability**: Modular and extensible design for future enhancements, also can be attached with Apache Airflow, and Apache Kafka to automate the extraction and streaming process along with AWS.

---

## Getting Started  

### Prerequisites  
- Python 3.12+  
- Required libraries: `pandas`, `numpy`, `sentence-transformers`, `scikit-learn`, `webcolors`, `selenium`, etc.  
- Install dependencies using:  
  ```bash
  pip install -r requirements.txt
