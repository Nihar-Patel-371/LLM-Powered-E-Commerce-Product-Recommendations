import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import webcolors
import matplotlib

# Function to read and preprocess data
def read_datasets_and_filter_col(path='handm.pkl'):
    # Read DB
    df = pd.read_pickle(path)
    columns_to_keep = ['productName', 'brandName', 'price', 'newArrival', 'colorName', 'colors', 'colorShades', 'mainCatCode', 'details', 'materials']
    return df[columns_to_keep]


def get_closest_color_name(rgb_tuple):
    min_distance = float('inf')
    closest_color = None
    for c_name, c_code in matplotlib.colors.cnames.items():
        r, g, b = webcolors.hex_to_rgb(c_code)
        distance = color_distance(rgb_tuple, (r, g, b))
        if distance < min_distance:
            min_distance = distance
            closest_color = c_name
    return closest_color

def color_distance(color1, color2):
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return (r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2

def preprocess_data(data):
    # productName ******************************************
    # Replace NaN values and Lower the case for product name
    data['productName'] = data['productName'].fillna('no product name')
    data['productName'] = [x.lower() for x in data['productName']]
    # brandName ********************************************
    # Replace NaN values and Lower the case for product name
    data['brandName'] = data['brandName'].fillna('no product brand')
    data['brandName'] = [x.lower() for x in data['brandName']]
    # price ************************************************
    # Preprocess numerical features
    data['price'] = pd.to_numeric(data['price'], errors='coerce').fillna(0.0)
    # newArrival *******************************************
    data['newArrival'] = data['newArrival'].astype(int)
    # colorName, colors, colorShades ***********************
    # Preprocess color based categorical fields
    categorical_cols = ['colorName', 'colorShades', 'mainCatCode']
    for col in categorical_cols:
        data[col] = data[col].fillna('unknown').astype(str)

    color_shades = []
    for color_code in data['colors']:
        if color_code!='unknown':
            rgb_tuple = webcolors.hex_to_rgb("#"+color_code)
            color_name = get_closest_color_name(rgb_tuple)
        else:
            color_name = 'unknown'
        color_shades.append(color_name)
    data['colors'] = color_shades
    # mainCatCode ******************************************
    data['mainCatCode'] = [x.replace('_', ' ') for x in data['mainCatCode']]
    # description ******************************************
    data['description'] = data['details'] + data['materials']
    # Replace NaN values and Lower the case for description
    data['description'] = data['description'].fillna('No description')
    data['description'] = [x.lower() for x in data['description']]
    # Create structured feature matrix
    numerical_fields = ['price', 'newArrival']
    categorical_fields = ['productName', 'brandName', 'colorName', 'colors', 'colorShades', 'mainCatCode', 'description']
    return data[numerical_fields], data[categorical_fields]

def combine_text(data):

    # Columns descriptions (meta-data)
    column_descriptions = {
        'productName': "The name of the product: ",
        'brandName': "The name of the brand manufacturing the product: ",
        'price': "The cost of the product in US dollars: ",
        'stockState': "Is the product is currently in stock: ",
        'comingSoon': "Shows if the product is scheduled for a future release",
        'isOnline': "Is the product available for online purchase: ",
        'newArrival': "Highlights whether the product is a newly released item: ",
        'colorName': "The primary color of the product: ",
        'colors': "List of colors associated with the product: ",
        'colorShades': "Tone of the product's color",
        'mainCatCode': "Main product catagory: ",
        'description': "Description of the product's features and qualities: "
    }

    # List of text fields to aggregate
    text_fields = data.columns

    # Function to combine each row with column descriptions
    def combine_with_descriptions(row):
        combined_text = [f"{column_descriptions[col]} {row[col]}" if col in column_descriptions else str(row[col]) for col in text_fields]
        return " | ".join(combined_text)

    # Apply the function to create the combined text
    data['combined_text'] = data.apply(combine_with_descriptions, axis=1)

    return data['combined_text']

def compute_embeddings(texts, model, tokenizer, device, batch_size=64):
    embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

if __name__ == '__main__':

    # Read and preprocess data
    data = read_datasets_and_filter_col(path='handm.pkl')
    num_data, col_data = preprocess_data(data)
    print(num_data, col_data)
    col_data = pd.concat([num_data, col_data], axis=1)
    combined_text = combine_text(col_data)
    print("*********************************\n", combined_text.tolist()[0])

    # Load model and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name =  "sentence-transformers/all-mpnet-base-v2" # "sentence-transformers/all-MiniLM-L6-v2", "google-bert/bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Compute embeddings
    text_embeddings = compute_embeddings(combined_text.tolist(), model, tokenizer, device)
    print("Text Embeddings Shape:", text_embeddings.shape)

    # Save embeddings and data
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump((text_embeddings, data), file)
    print("Model training complete and saved to 'trained_model.pkl'.")