# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 


# %%
game = pd.read_csv("game.csv")
game


# %%
game.isna().sum()

# %% [markdown]
# ## No Missing Files

# %%
# Creating a Tfidf Vectorizer to remove all stop words

tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer
tfidf
# taking top english top words


# %%
# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(game.game)  

 # Transform a count matrix to a normalized tf or tf-idf representation


tfidf_matrix.shape 


# %%
tfidf_matrix

# %% [markdown]
# with the above matrix we need to find the similarity score¶
# 
# There are several metrics for this such as the euclidean,
# 
# the Pearson and the cosine similarity scores
# 
# For now we will be using cosine similarity matrix
# 
# A numeric quantity to represent the similarity between 2 Games
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

gamee_index = pd.Series(game.index, index = game['game']).drop_duplicates()
gamee_index


# %%
def get_recommendations(game, topN):   

    # topN = 10
    # Getting the game index using its title 

    game_id = gamee_index[game]
    
    # Getting the pair wise similarity score for all the entertainment's with that entertainment

    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # Sorting the cosine_similarity scores based on scores 

    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 

    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 

    game_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores

    game_similar_show = pd.DataFrame(columns=["Score"])
    game_similar_show["game"] = game.loc[game_idx, "game"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)  

    print (game_similar_show)
    
   


# %%
get_recommendations("The Legend of Zelda: Ocarina of Time", topN = 10)
gamee_index["The Legend of Zelda: Ocarina of Time"]


# %%



