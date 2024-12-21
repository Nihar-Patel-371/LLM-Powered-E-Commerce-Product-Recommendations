# LLM-Powered-E-Commerce-Product-Recommendations

## Overview  
This repository contains a project focused on building an end-to-end web mining and recommendation system. The system collects product data from various e-commerce websites, processes it, and uses advanced modeling techniques to recommend products based on user queries.  

Key features include:  
- **Web Scraping**: Extracts structured and unstructured product data.  
- **Data Processing**: Prepares and refines data through feature engineering and cleaning.  
- **Sentence Transformers**: Embeds textual data into meaningful vector representations using models like `all-mpnet-base-v2` and `all-Mini-L6-v2`.  
- **Cosine Similarity**: Matches user queries with products based on vector similarity for recommendations.  
- **Scalability**: Modular and extensible design for future enhancements.

---

## Key Highlights  
- **Data Collection**: Leverages APIs, Selenium, and web scraping to extract product details.  
- **Preprocessing**: Converts HEX color codes to RGB and clusters products by categories for richer features.  
- **Modeling**: Uses Sentence Transformers for vector embeddings with detailed inference.  
- **Recommendations**: Cosine similarity-based approach for identifying the most relevant products.  

---

## Applications  
This project is tailored for:  
- E-commerce platforms to deliver personalized product recommendations.  
- Data-driven analysis of online product trends.  
- Enhancing user experience by matching queries to precise results.  

---

## Getting Started  

### Prerequisites  
- Python 3.12+  
- Required libraries: `pandas`, `numpy`, `sentence-transformers`, `scikit-learn`, `webcolors`, `selenium`, etc.  
- Install dependencies using:  
  ```bash
  pip install -r requirements.txt
