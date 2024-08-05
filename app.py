import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from fuzzywuzzy import process

# Define the API key (for TheMealDB, the test key is "1")
api_key = "1"

# Fetch recipe data using TheMealDB API
def fetch_recipes(api_key, letter='a'):
    url = f"https://www.themealdb.com/api/json/v1/{api_key}/search.php?f={letter}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('meals', [])
    else:
        st.error(f"Error fetching recipes for letter {letter}: {response.status_code}")
        return []

# Fetch details for all recipes starting with letters 'a' to 'z'
all_recipes = []
for letter in 'abcdefghijklmnopqrstuvwxyz':
    recipes = fetch_recipes(api_key, letter=letter)
    if recipes:  # Only extend if there are recipes returned
        all_recipes.extend(recipes)

# Check if any recipes were fetched
if not all_recipes:
    st.error("No recipes were fetched. Please check the API response or try again later.")
else:
    df_recipes = pd.DataFrame(all_recipes)

    # Display the basic structure
    st.write(df_recipes.head())

    if 'idMeal' not in df_recipes.columns:
        st.error("Error: 'idMeal' column not found in the fetched data.")
        st.write("Columns present:", df_recipes.columns)

# Rename columns for consistency
df_recipes.rename(columns={
    'idMeal': 'id', 'strMeal': 'title', 'strCategory': 'category',
    'strArea': 'area', 'strInstructions': 'instructions',
    'strMealThumb': 'image', 'strTags': 'tags'
}, inplace=True)

# Function to extract ingredients from TheMealDB format
def extract_ingredients(row):
    ingredients = []
    for i in range(1, 21):
        ingredient = row[f'strIngredient{i}']
        if ingredient and ingredient.strip():
            ingredients.append(ingredient.strip())
    return ingredients

# Extract ingredients and clean data
df_recipes['ingredients'] = df_recipes.apply(extract_ingredients, axis=1)
df_recipes = df_recipes.drop(columns=[f'strIngredient{i}' for i in range(1, 21)])
df_recipes_clean = df_recipes[[
    'id', 'title', 'category', 'area', 'instructions', 'tags', 'image', 'ingredients'
]]

# Display available columns and basic data
st.write(df_recipes_clean.columns)
st.write(df_recipes_clean.head())

# Combine textual features
df_recipes_clean['combined_features'] = df_recipes_clean.apply(
    lambda x: f"{x['title']} {' '.join(x['ingredients'])} {x['category']} {x['area']} {x['tags'] if x['tags'] else ''}", axis=1
)

# Vectorize the combined features
vectorizer = TfidfVectorizer(stop_words='english')
features_matrix = vectorizer.fit_transform(df_recipes_clean['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(features_matrix, features_matrix)

# Function to normalize a single title
def normalize_title(title):
    return title.lower().strip()

# Function to suggest similar titles
def suggest_similar_titles(title, df, threshold=60):
    normalized_title = normalize_title(title)
    titles = df['title'].tolist()
    matches = process.extract(normalized_title, titles, limit=5)
    return [match for match in matches if match[1] >= threshold]

# Function to get content-based recommendations with fallback suggestions
def get_content_based_recommendations(title, cosine_sim, df, top_n=5):
    normalized_title = normalize_title(title)
    if normalized_title not in df['title'].values:
        suggestions = suggest_similar_titles(normalized_title, df, threshold=50)
        st.write(f"Recipe title '{title}' not found in the dataset.")
        if suggestions:
            st.write("Did you mean one of these?")
            for suggestion in suggestions:
                st.write(f" - {suggestion[0]}")
        return pd.DataFrame()  # Return an empty DataFrame

    # Proceed with finding the recommendations
    idx = df[df['title'] == normalized_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recipe_indices = [i[0] for i in sim_scores]
    return df.iloc[recipe_indices][['title', 'category', 'area', 'tags', 'ingredients']]

# UI elements for user input
st.title("Recipe Recommendation System")
title = st.text_input("Enter a recipe title")

# Display recommendations
if title:
    recommendations = get_content_based_recommendations(title, cosine_sim, df_recipes_clean)
    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            st.image(row['image'], width=100)
            st.markdown(f"**{row['title']}**")
            st.write(f"Category: {row['category']}")
            st.write(f"Area: {row['area']}")
            st.write(f"Ingredients: {', '.join(row['ingredients'])}")
            st.write("---")
    else:
        st.write("No recommendations available.")
