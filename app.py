import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        print(f"Error fetching recipes for letter {letter}: {response.status_code}")
        return []

# Fetch details for all recipes starting with letters 'a' to 'z'
all_recipes = []
for letter in 'abcdefghijklmnopqrstuvwxyz':
    recipes = fetch_recipes(api_key, letter=letter)
    if recipes:
        all_recipes.extend(recipes)

df_recipes = pd.DataFrame(all_recipes)
df_recipes.rename(columns={
    'idMeal': 'id', 'strMeal': 'title', 'strCategory': 'category',
    'strArea': 'area', 'strInstructions': 'instructions',
    'strMealThumb': 'image', 'strTags': 'tags'
}, inplace=True)

def extract_ingredients(row):
    ingredients = []
    for i in range(1, 21):
        ingredient = row[f'strIngredient{i}']
        if ingredient and ingredient.strip():
            ingredients.append(ingredient.strip())
    return ingredients

df_recipes['ingredients'] = df_recipes.apply(extract_ingredients, axis=1)
df_recipes = df_recipes.drop(columns=[f'strIngredient{i}' for i in range(1, 21)])
df_recipes_clean = df_recipes[[
    'id', 'title', 'category', 'area', 'instructions', 'tags', 'image', 'ingredients'
]]

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
    return df.iloc[recipe_indices][['title', 'category', 'area', 'tags', 'ingredients', 'image']]

# Streamlit UI
st.title("Recipe Recommendation System")
st.write("Get Recipe Recommendations")

input_title = st.text_input("Enter a recipe title", "Pasta")
recommendations = get_content_based_recommendations(input_title, cosine_sim, df_recipes_clean)

if not recommendations.empty:
    st.write("Recommended Recipes:")
    for _, row in recommendations.iterrows():
        st.write(f"### {row['title']}")
        st.write(f"Category: {row['category']}")
        st.write(f"Area: {row['area']}")
        st.write(f"Tags: {row['tags']}")
        st.write(f"Ingredients: {', '.join(row['ingredients'])}")
        if row['image']:
            st.image(row['image'], width=200)
        st.write("---")
