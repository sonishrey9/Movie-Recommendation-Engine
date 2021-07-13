# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 

# %% [markdown]
# ### term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

# %%
entertainment = pd.read_csv("Entertainment.csv")
entertainment


# %%
entertainment.isna().sum()

# %% [markdown]
# ## No Missing Files

# %%
# Creating a Tfidf Vectorizer to remove all stop words

tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer

# taking top english top words


# %%
# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(entertainment.Category)  

 # Transform a count matrix to a normalized tf or tf-idf representation

tfidf_matrix.shape #12294, 46

# %% [markdown]
# with the above matrix we need to find the similarity score¶
# 
# There are several metrics for this such as the euclidean,
# 
# the Pearson and the cosine similarity scores
# 
# For now we will be using cosine similarity matrix
# 
# A numeric quantity to represent the similarity between 2 movies
# 
# Cosine similarity - metric is independent of magnitude and easy to calculate

# %%
from sklearn.metrics.pairwise import linear_kernel


# %%
# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim_matrix


# %%
# creating a mapping of entertainment name to index number 

entertainment_index = pd.Series(entertainment.index, index = entertainment['Titles']).drop_duplicates()
entertainment_index


# %%
def get_recommendations(Titles, topN):   

    # topN = 10
    # Getting the movie index using its title 

    entertainment_id = entertainment_index[Titles]
    
    # Getting the pair wise similarity score for all the entertainment's with that entertainment

    cosine_scores = list(enumerate(cosine_sim_matrix[entertainment_id]))
    
    # Sorting the cosine_similarity scores based on scores 

    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 

    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 

    entertainment_idx  =  [i[0] for i in cosine_scores_N]
    entertainment_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores

    entertainment_similar_show = pd.DataFrame(columns=["Score"])
    entertainment_similar_show["Titles"] = entertainment.loc[entertainment_idx, "Titles"]
    entertainment_similar_show["Score"] = entertainment_scores
    entertainment_similar_show.reset_index(inplace = True)  

    print (entertainment_similar_show)
    
   


# %%
# Enter your anime and number of anime's to be recommended 
get_recommendations("Clueless (1995)", topN = 10)
entertainment_index["Clueless (1995)"]


# %%



