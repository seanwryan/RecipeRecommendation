!pip install pandas numpy requests scikit-learn nbconvert fuzzywuzzy[speedup]

# Import necessary libraries
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
        print(f"Error fetching recipes for letter {letter}: {response.status_code}")
        return []

# Fetch details for all recipes starting with letters 'a' to 'z'
all_recipes = []
for letter in 'abcdefghijklmnopqrstuvwxyz':
    recipes = fetch_recipes(api_key, letter=letter)
    if recipes:  # Only extend if there are recipes returned
        all_recipes.extend(recipes)

# Check if any recipes were fetched
if not all_recipes:
    print("No recipes were fetched. Please check the API response or try again later.")
else:
    df_recipes = pd.DataFrame(all_recipes)

    # Display the basic structure
    print(df_recipes.head())

    if 'idMeal' not in df_recipes.columns:
        print("Error: 'idMeal' column not found in the fetched data.")
        print("Columns present:", df_recipes.columns)

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
print(df_recipes_clean.columns)
print(df_recipes_clean.head())

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
        print(f"Recipe title '{title}' not found in the dataset.")
        if suggestions:
            print("Did you mean one of these?")
            for suggestion in suggestions:
                print(f" - {suggestion[0]} (Confidence: {suggestion[1]}%)")
        return pd.DataFrame()  # Return an empty DataFrame

    # Proceed with finding the recommendations
    idx = df[df['title'] == normalized_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    recipe_indices = [i[0] for i in sim_scores]
    return df.iloc[recipe_indices][['title', 'category', 'area', 'tags', 'ingredients']]

# Example usage: Get recommendations for a specific recipe title
recommendations = get_content_based_recommendations('Arrabiata', cosine_sim, df_recipes_clean)
print(recommendations)

# Simulate user-recipe interaction matrix for collaborative filtering
np.random.seed(42)
user_recipe_matrix = np.random.randint(2, size=(10, 10))

# Apply SVD to the user-recipe matrix
svd = TruncatedSVD(n_components=5, random_state=42)
latent_matrix = svd.fit_transform(user_recipe_matrix)
latent_matrix = np.dot(latent_matrix, svd.components_)

# Function to get user-based recommendations
def get_user_recommendations(user_index, latent_matrix=latent_matrix):
    user_prefs = latent_matrix[user_index]
    recipe_indices = np.argsort(user_prefs)[::-1]
    return df_recipes_clean.iloc[recipe_indices[:5]]

# Example usage: Get recommendations for a user
user_recommendations = get_user_recommendations(0)
print(user_recommendations)
