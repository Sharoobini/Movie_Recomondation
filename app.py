import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# APP TITLE & DESCRIPTION
# -------------------------------
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System ")
st.write("""
This app demonstrates a **Movie Recommendation System** built using the 
**MovieLens Kaggle dataset (movies.csv)** and **content-based filtering** with TF-IDF on genres.

You can:
- Explore the dataset  
- Visualize ratings distributions  
- Generate similar-movie recommendations  
- View evaluation placeholders for model performance
""")

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
menu = st.sidebar.radio(
    "📌 Navigation",
    ["Dataset Exploration", "Visualisations", "Model Prediction", "Model Performance"]
)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    return movies

movies = load_data()

# -------------------------------
# CONTENT-BASED SIMILARITY
# -------------------------------
@st.cache_resource
def build_similarity_matrix(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_similarity_matrix(movies)

def recommend_movies(title, n=5):
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    if title not in indices:
        return pd.DataFrame(columns=["Title","Genres","Similarity"])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    recs = movies.iloc[movie_indices][['title','genres']].copy()
    recs['similarity'] = scores
    return recs

# -------------------------------
# DATASET EXPLORATION
# -------------------------------
if menu == "Dataset Exploration":
    st.header("📊 Dataset Overview")
    
    st.write("### Movies Dataset")
    st.write(f"Shape: {movies.shape}")
    st.write(movies.dtypes)
    st.dataframe(movies.head())

    st.write("### 🔍 Search/Filter Movies")
    search = st.text_input("Search by movie title")
    if search:
        st.write(movies[movies['title'].str.contains(search, case=False, na=False)].head(10))

# -------------------------------
# VISUALISATIONS
# -------------------------------
elif menu == "Visualisations":
    st.header("📈 Visualisations")
    
    # 1. Movie release year distribution
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    fig, ax = plt.subplots()
    movies['year'].dropna().astype(int).hist(bins=30, ax=ax)
    ax.set_title("Distribution of Movies by Year")
    st.pyplot(fig)

    # 2. Genre counts
    genre_counts = movies['genres'].str.split('|').explode().value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax)
    ax.set_title("Top Genres")
    st.pyplot(fig)

    # 3. Movies per decade
    movies['decade'] = (movies['year'].dropna().astype(float)//10*10).astype('Int64')
    decade_counts = movies['decade'].value_counts().sort_index()
    fig, ax = plt.subplots()
    decade_counts.plot(kind='bar', ax=ax)
    ax.set_title("Movies per Decade")
    st.pyplot(fig)

# -------------------------------
# MODEL PREDICTION (Content-based recs)
# -------------------------------
elif menu == "Model Prediction":
    st.header("🎯 Movie Recommendations (Content-based)")
    st.write("Enter a movie title to get similar recommendations:")

    title = st.text_input("Movie Title", "Toy Story (1995)")
    n = st.slider("Number of Recommendations", 1, 20, 5)

    if st.button("Recommend"):
        recs = recommend_movies(title, n=n)
        st.dataframe(recs)

# -------------------------------
# MODEL PERFORMANCE (placeholder)
# -------------------------------
elif menu == "Model Performance":
    st.header("📊 Model Performance Metrics")
    st.write("""
    Since this is a **content-based model**, we don't have traditional accuracy metrics 
    (like RMSE or MAE) without ratings data.  
    Instead, evaluation is often done with user studies or offline ranking metrics.
    """)

    # Example placeholder confusion-like matrix (not real for this case)
    y_true = [1, 0, 1, 1, 0, 1]  # fake ground truth
    y_pred = [1, 0, 1, 0, 0, 1]  # fake predictions

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Not Relevant","Relevant"]).plot(ax=ax)
    st.pyplot(fig)

    st.write("### Model Comparison (placeholder)")
    st.write("""
    In practice, you could compare:
    - Content-based similarity (genres, tags, text)  
    - Collaborative filtering (if ratings.csv available)  
    - Hybrid models  
    """)
