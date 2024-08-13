import pickle
import lzma
from random import Random
import concurrent.futures
import datetime
from Practical2 import *
import os

def predict(n_user,model_name):
    DIR = os.getcwd()+ "/model/"+model_name

    if not (os.path.isfile(DIR)):
        raise Exception("Loading Error: model does not exit")
    
    model = read_embeddings(DIR)

    [user_baises, item_biases] = [model['user_biases'], model['item_biases']]
    [user_vec, item_vec] = [model['user_vec'], model['item_vec']]

    lam = model['lambda']
    gamma = model['gamma']
    tau = model['tau']
    k = model['k']

    u_vec = embed_new_user(n_user,[item_biases, item_vec],gamma,lam,tau,k=k)
    u_rating = np.dot(u_vec, item_vec.T) + 0.3 * item_biases 
    
    return u_rating

def embed_new_user(ur,item,gamma,lam,tau,N = 50,k=10,feature_vec=[]):
    item_biases, item_vec = item
    user_baises = []

    user_vec = np.zeros((1,k))
    
    for i in tqdm(range(N), desc="Epoch"):

        user_ratings = np.array([r for (_, r) in ur])
        item_indices = np.array([n for (n, _) in ur])
        item_biases_m = item_biases[item_indices]
        item_count = len(item_indices )
       
        user_biases = np.sum(lam * (user_ratings - np.dot(user_vec, item_vec[item_indices].T)- item_biases_m)) / ((lam * item_count) + gamma)
        
        # Compute item vector update
        v_n = np.zeros((k, k))
        vr_n = np.zeros(k)

        for n, r in ur:
            item_vec_n = item_vec[n].reshape(-1, 1)
            v_n += np.dot(item_vec_n, item_vec_n.T) + tau * np.eye(k)
            vr_n += item_vec[n] * (r - user_biases - item_biases[n])

        user_vec = np.linalg.inv(v_n) @ vr_n

    return user_vec

def get_top_10(rating_vec,item,movie_data='ml-25m'):
    #get save data
    # path = os.getcwd()+'/DATA/ml-latest-small/movie_data.xz'
    path = os.getcwd()+f'/DATA/{movie_data}/movie_data.xz'
    movie_data = read_embeddings(path)
    popularity = movie_data['movie_popularity']
    title = movie_data['title']
    genres = movie_data['genres']
    movie_to_idx ={ m_id:idx for idx, m_id in enumerate( movie_data['movieId'])}

    #ranking movies 
    rank = np.argsort(rating_vec)[::-1]

    #get popularity of the highest ranked moveds4993
    popularity_on_rank = [ i for i in rank if popularity[i] > 50]

    print("\n=================== Movie Seen ===========================")
    idx = movie_to_idx[item]
    print(title[idx],f"| Genre : {genres[idx]}")
    print("\n=================== Recommended movies ===================")
    for i in popularity_on_rank[:20]:
        print(title[i],f"| Genre : {genres[i]}")

def search_movie_title(query, df):
    """
    Search for movies whose titles match part .
    
    Parameters:
    query (str): The string to search for in movie titles.
    df (pd.DataFrame): The DataFrame containing movie data.
    
    Returns:
    pd.DataFrame: A DataFrame with matching movies.
    """
    # Use the 'str.contains' method for case-insensitive search
    result_df = df[df['title'].str.contains(query, case=False, na=False)]
    return result_df

if __name__ == "__main__":
    path = os.getcwd()+'/DATA/ml-25m/movie_data.xz'
    movie_data = read_embeddings(path)
    popularity = movie_data['movie_popularity']
    title = movie_data['title']
    genres = movie_data['genres']
    movie_to_idx ={ m_id:idx for idx, m_id in enumerate( movie_data['movieId'])}


    # new_user = [(movie_to_idx[5349],5)]
    # new_user = [(movie_to_idx[4993],0.5),(34,2)]
    new_user = [(movie_to_idx[136343],5)]
    # new_user = [(movie_to_idx[493],5)]
    vec = predict(new_user,"feat_mdl_ltkg_202024-05-11-21:37:14.xz")

    get_top_10(vec,136343)