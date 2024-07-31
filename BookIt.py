import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

st.header('BookIt: Book Rentals Made Easy!')

hometab, searchtab = st.tabs(['Home', 'Search'])

# Load the books data
books = pd.read_csv("C:\\Users\\josea\\OneDrive\\Desktop\\Feynn ML Internship\\Project 3\\Books.csv", encoding = 'latin-1')

# Load the ratings data
ratings = pd.read_excel("C:\\Users\\josea\\OneDrive\\Desktop\\Feynn ML Internship\\Project 3\\Ratings.xlsx")

# Calculating the average rating of each book
books = pd.merge(books, pd.DataFrame(ratings.groupby('BID').mean()['Rating'], columns = ['Rating']), on = 'BID') 
books['Rating'] = books['Rating'].apply(lambda x: round(x, 1))

# Load the users data
users = pd.read_csv("C:\\Users\\josea\\OneDrive\\Desktop\\Feynn ML Internship\\Project 3\\Users.csv")

def collaborative_filtering(ratings, user_preferences, books, num_recommendations = 10):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['UID', 'BID', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd = SVD()
    svd.fit(trainset)
    
    # Generate a new temporary user ID
    new_user_id = ratings['UID'].max() + 1
    
    # Predict ratings for all books for the new user
    book_ids = books['BID'].unique()
    user_ratings = {}
    for book_id in book_ids:
        pred = svd.predict(new_user_id, book_id)
        user_ratings[book_id] = pred.est
    
    # Get the top N recommendations
    recommended_books_ids = sorted(user_ratings, key=user_ratings.get, reverse=True)[:num_recommendations]
    return recommended_books_ids

def content_based_filtering(user_preferences, books, num_recommendations = 10):
    categories = books['Category'].unique()
    category_matrix = pd.get_dummies(books['Category'])
    
    # Align user preferences with the categories
    category_indices = {category: i for i, category in enumerate(categories)}
    user_profile = np.zeros(len(categories))
    
    for category, rating in zip(categories, user_preferences):
        if category in category_indices:
            user_profile[category_indices[category]] = rating
    
    # Calculate the similarity between user profile and book categories
    category_similarity = cosine_similarity(user_profile.reshape(1, -1), category_matrix)
    recommended_books = books.iloc[np.argsort(-category_similarity).flatten()][:num_recommendations]
    return recommended_books['BID'].tolist()

def get_recommendations(user_preferences, books, ratings, num_recommendations = 10):
    content_recommendations = content_based_filtering(user_preferences, books, num_recommendations)
    collaborative_recommendations = collaborative_filtering(ratings, user_preferences, books, num_recommendations)
    
    # Combine recommendations, giving weight to collaborative filtering
    combined_recommendations = content_recommendations[:num_recommendations//2] + collaborative_recommendations[:num_recommendations//2]
    
    # Remove duplicates while maintaining order
    combined_recommendations = list(dict.fromkeys(combined_recommendations))
    
    recommended_books = books[books['BID'].isin(combined_recommendations)]
    return recommended_books

def get_user_preferences():
    st.header("Rate Your Preferences")
    categories = books['Category'].unique()
    num_columns = 4
    user_preferences = []

    columns = st.columns(num_columns)
    for i, category in enumerate(categories):
        col = columns[i % num_columns]
        with col:
            rating = st.slider(f'Rate your interest in {category}', 1, 5, 1, key = category)
            user_preferences.append(rating)
    
    return np.array(user_preferences), categories

with hometab:
    if 'show_recommendations' not in st.session_state:
        st.session_state['show_recommendations'] = False

    if not st.session_state['show_recommendations']:
        user_preferences, categories = get_user_preferences()
        if st.button("Get Recommendations"):
            st.session_state['user_preferences'] = user_preferences
            st.session_state['categories'] = categories
            st.session_state['show_recommendations'] = True
            st.rerun()
    else:
        user_preferences = st.session_state['user_preferences']
        categories = st.session_state['categories']
        recommendations = get_recommendations(user_preferences, books, ratings)
        
        st.header("Recommended Books for You")
        num_columns = 4
        columns = st.columns(num_columns)
        for i, (_, row) in enumerate(recommendations.iterrows()):
            col = columns[i % num_columns]
            with col:
                st.write(f"**Title:** {row['Title']}")
                st.write(f"**Author:** {row['Author']}")
                st.write(f"**Category:** {row['Category']}")
                st.markdown("***")
        
        if st.button("Refresh"):
            st.session_state['show_recommendations'] = False
            st.rerun()
                
with searchtab:
    # Search bar
    search_query = st.text_input("Search for a book: ")

    if search_query:
        # Filter books based on the search query
        search_results = books[books['Title'].str.contains(search_query, na = False)]

        # Create a list of search result titles
        search_titles = search_results['Title'].tolist()

        if search_titles:
            # Dropdown with search results
            selected_title = st.selectbox("Select a book from search results:", search_titles)
            # Display the selected book details    
            selected_book = search_results[search_results['Title'] == selected_title]
            cols = st.columns(5)
            with cols[0]:
                st.metric(label = "**Author**: ", value = selected_book['Author'].values[0])
            with cols[1]:
                st.metric(label = "**Category**: ", value = selected_book['Category'].values[0])
            with cols[2]:
                st.metric(label = "**Status**: ", value = selected_book['Status'].values[0])
            with cols[3]:
                st.metric(label = "**Rating**: ", value = round(selected_book['Rating'].values[0], 1))
            with cols[4]:
                st.metric(label = "**Available At**: ", value = selected_book['Library'].values[0])
        else:
            st.write("No results to display.")
    else:
        st.dataframe(books.set_index('Title').iloc[:, 1:], use_container_width = True, height = int(35.62*len(books)))
