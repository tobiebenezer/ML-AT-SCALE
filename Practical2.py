import requests
from zipfile import ZipFile 
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.stats import halfnorm
import json
import pickle
import lzma
from random import Random
import concurrent.futures
import datetime

compress_filename = "data_embedings.xz"
#download and extract data
def load_data(url,filename):
    DIR = os.getcwd()

    if not (os.path.isdir(f"{DIR}/DATA")):
        os.mkdir(f"{DIR}/DATA")

    response = requests.get(url, stream=True)
    with open(F"{DIR}/DATA/{filename}", mode="wb") as file:
        file.write(response.content)

    unzip_file(F"{DIR}/DATA/{filename}",F"{DIR}/DATA/")

def unzip_file(path,dest_path):
    # loading the temp.zip and creating a zip object 
    with ZipFile(path, 'r') as zipObject: 
        # Extracting all the members of the zip 
        zipObject.extractall(path=f"{dest_path}") 


# load_data("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip","ml-latest-small.zip")
# load_data("https://files.grouplens.org/datasets/movielens/ml-25m.zip",'ml-25m.zip')

def get_data(folder):
    DIR = os.getcwd()
    data_path = f"{DIR}/DATA/{folder}"

    # Creating embedings if it yet to be saved in memeory
    if not (os.path.isfile(f"{data_path}/{compress_filename}")):
        
        movies_df = pd.read_csv(f'{data_path}/movies.csv')
        ratings_df = pd.read_csv(f'{data_path}/ratings.csv')
        # tags_df = pd.read_csv(f'{data_path}/tags.csv')

        rating_arr = ratings_df[['userId','movieId','rating']].to_numpy()
        train , test, unique_user_ids, unique_movie_ids = train_test_split(rating_arr)

        # compute movie popularity
        popularity = compute_movie_popularity(ratings_df,unique_movie_ids)

        # get movie feature embeding and titles
        [feature_one_hot_to_index ,
         titles_to_index , genre_to_index, genre_list ] =  movie_to_genres(movies_df,unique_movie_ids)
        
        movie_data = {
            'feature_one_hot' : feature_one_hot_to_index,
            'title':titles_to_index,
            'genres': genre_to_index,
            'movieId': unique_movie_ids,
            'movie_popularity' :popularity,
            'genre_list':genre_list,
        }

        #create Data embdings for user and items
        train_uv = create_UV(train,unique_user_ids, unique_movie_ids)
        test_uv = create_UV(test, unique_user_ids, unique_movie_ids)
        
        #saving embeddings
        data = save_embedings(train_uv,test_uv,data_path)
        save_data(movie_data,data_path+"/movie_data")

        train_uv = data['train']
        test_uv = data['test']
    else:
        #user embeding and item embedding
        data = read_embeddings(f"{data_path}/{compress_filename}")
        train_uv = data['train']
        test_uv = data['test']

        #movie data
        movie_data = read_embeddings(data_path+"/movie_data.xz")
        genre_list = movie_data['genre_list']
        
        feature_one_hot_to_index = movie_data['feature_one_hot']
        
    return train_uv, test_uv,[genre_list ,feature_one_hot_to_index] 

def create_UV(rating_arr,unique_user_ids,unique_movie_ids):
    
    #sorting according to user and movies
    user_sort = rating_arr[rating_arr[:, 0].argsort()]
    movies_sort = rating_arr[rating_arr[:, 1].argsort()]

    # separating user and movie IDs and ratings
    user_ids, user_movie_ids, user_ratings = user_sort.T
    movie_user_ids, movie_movie_ids, movie_ratings = movies_sort.T

    # mapping Ids to rating map
    user_to_idx = {int(user_id): idx for idx, user_id in enumerate(unique_user_ids)}
    movie_to_idx = {int(movie_id): idx for idx, movie_id in enumerate(unique_movie_ids)}

    # mapping
    user_to_rating = [[] for _ in unique_user_ids]
    movie_to_rating = [[] for _ in unique_movie_ids]

    for user_id, movie_id, rating in tqdm(zip(user_ids, user_movie_ids, user_ratings)):
        user_idx = user_to_idx[int(user_id)]
        movie_idx = movie_to_idx[int(movie_id)]
        user_to_rating[user_idx].append((movie_idx, rating))
        movie_to_rating[movie_idx].append((user_idx, rating))

    return user_to_idx, movie_to_idx, user_to_rating, movie_to_rating

def movie_to_genres(movies_df,movie_to_idx):
    movies_df.loc[movies_df['genres'] == '(no genres listed)', 'genres'] = '<UNK/>'
    genres = np.unique([ gen_type for genre in movies_df['genres'].unique() for gen_type in genre.split('|')])

    # genre to index
    genre_to_index = {genre:idx for idx, genre in enumerate(genres)}
    
    def genre_to_one_hot(genre_string):
        one_hot = [0]*len(genres)
        for genre in genre_string.split('|'):
            one_hot[genre_to_index[genre]] = 1
        return one_hot

    # Apply the function to each row of the DataFrame and convert to numpy array
    movies_df['feature_one_hot'] = movies_df['genres'].apply(genre_to_one_hot)
    movies_df['idx'] = movies_df['movieId'].map({ mid:idx for idx,mid in enumerate(movie_to_idx)})
    movies_df=movies_df.dropna(subset=['idx'])

    #sorting by index
    movies_data = movies_df.to_numpy()
    movies_data = movies_data[movies_data[:, 4].argsort()]

    #get list for updated
    feature_one_hot_to_index  = [[] for _ in range(len(movies_data))]
    titles_to_index  = [[] for _ in range(len(movies_data))]
    genre_to_index = [[] for _ in range(len(movies_data))]

    #fill update list with values
    for i in range(len(movies_data)):
        _,title, genre_index, one_hot,_ = movies_data[i]
        feature_one_hot_to_index[i] = one_hot
        genre_to_index[i] = genre_index.split('|')
        titles_to_index[i] = title

    return [
        feature_one_hot_to_index ,
        titles_to_index ,
        genre_to_index ,
        genres
    ]

def save_embedings(train,test,data_path):
    train_user_to_idx, train_movie_to_idx, train_user_to_rating, train_movie_to_rating = train
    test_user_to_idx, test_movie_to_idx, test_user_to_rating, test_movie_to_rating = test

    data ={"train" : {
        "user_to_idx":train_user_to_idx, 
        "movie_to_idx":train_movie_to_idx, 
        "user_to_rating":train_user_to_rating, 
        "movie_to_rating":train_movie_to_rating
    },
    "test" : {
        "user_to_idx":test_user_to_idx, 
        "movie_to_idx":test_movie_to_idx, 
        "user_to_rating":test_user_to_rating, 
        "movie_to_rating":test_movie_to_rating
    }}

    with lzma.open(f'{data_path}/{compress_filename}', 'wb') as f:
        pickle.dump(data, f)
    
    return data

def save_model(data,path="m_"):
    DIR = os.getcwd()

    if not (os.path.isdir(f"{DIR}/model")):
        os.mkdir(f"{DIR}/model")

    tm = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    path = f'{DIR}/model/{path}{tm}'

    save_data(data,path)
        
def save_data(data,path):
    with lzma.open(f'{path}.xz', 'wb') as f:
        pickle.dump(data, f)

def read_embeddings(path):
    data={}
    with lzma.open(path, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        data = unpickler.load()

    return data

def compute_movie_popularity(df,movie_to_index):

    #popularity count
    ratings_count = df.groupby('movieId').size().reset_index(name='ratings_count')

    # Convert the result into a dictionary
    popularity_dict = dict(zip(ratings_count['movieId'], ratings_count['ratings_count']))
    popularity = [popularity_dict[movieId] for movieId in movie_to_index]

    return popularity

def train_test_split(data,test=0.2,seed=43):
    user_ids, movie_ids, user_ratings = data.T

    # getting unique user IDs and movie IDs
    unique_user_ids = np.unique(user_ids)
    unique_movie_ids = np.unique(movie_ids)
    
    # Making Data random
    rng = np.random.default_rng(43)
    rng.shuffle(data)
    
    test_num = int(test * len(data))

    test = data[:test_num+1, :]
    train = data[test_num+1:, :]

    return train, test, unique_user_ids, unique_movie_ids

def alt_least_sqrt_biaOnly(train,test,biases, latent_vec ,lam = 2e-3, gamma = 3e-2, tau=1e-3,k=5):
    U,V = train
    test_U, test_V = test

    M, N = len(U), len(V)
    
    user_biases, item_biases = biases
    user_vec, item_vec = latent_vec


    for m in range(M):
        bias = 0
        item_count = 0
        for n,r in U[m]:
            bias += lam * ( r - item_biases[n])
            item_count +=1

        bias = bias / ((lam * item_count) + gamma)
        user_biases[m] = bias

    for n in range(N):
        bias = 0
        user_count = 0
  
        for m,r in V[n]:
            bias += lam * ( r - user_biases[m])
            user_count += 1

        bias = bias /((lam * user_count) + gamma)
        item_biases[n] = bias

    return ([user_biases, item_biases],latent_vec)

def alt_least_sqrt(train,test,biases, latent_vec ,lam = 2e-3, gamma = 3e-2, tau=1e-3,k=5):
    U,V = train
    test_U, test_V = test

    M, N = len(U), len(V)
    
    user_biases, item_biases = biases
    user_vec, item_vec = latent_vec

    for m in range(M):
        if len(U[m]) > 0:
        # Calculate user bias
            user_ratings = np.array([r for (_, r) in U[m]])
            item_indices = np.array([n for (n, _) in U[m]])
            item_biases_m = item_biases[item_indices]
            item_count = len(item_indices)
            user_biases[m] = np.sum(lam * (user_ratings - np.dot(user_vec[m], item_vec[item_indices].T)- item_biases_m)) / ((lam * item_count) + gamma)
        
        # Compute item vector update
        v_n = np.zeros((k, k))
        vr_n = np.zeros(k)

        # for n, r in U[m]:
        #     item_vec_n = item_vec[n].reshape(-1, 1)
        #     v_n += np.dot(item_vec_n, item_vec_n.T) + tau * np.eye(k)
        #     vr_n += item_vec[n] * (r - user_biases[m] - item_biases[n])

        v_n = np.sum(np.einsum("ij,il->ijl",item_vec[item_indices],item_vec[item_indices]),axis=0) + tau * np.eye(k)
        vr_n = np.sum(np.einsum("ji,j->ji",item_vec[item_indices],(user_ratings - user_biases[m] -item_biases_m)), axis=0) 

        user_vec[m] = np.linalg.inv(v_n) @ vr_n


    for n in range(N):
        # Calculate item bias
        if len(V[n]) > 0:
            item_ratings = np.array([r for (_, r) in V[n]])
            user_indices = np.array([m for (m, _) in V[n]])
            user_biases_n = user_biases[user_indices]
            user_count = len(user_indices)
            item_biases[n] = np.sum(lam * (item_ratings - np.dot(user_vec[user_indices], item_vec[n].T) - user_biases_n)) / ((lam * user_count) + gamma)
        

        # Compute item vector update

        u_n =np.sum(np.einsum("ij,il->ijl",user_vec[user_indices],user_vec[user_indices]),axis=0) + tau * np.eye(k)
        ur_n = np.sum(np.einsum("ji,j->ji",user_vec[user_indices],(item_ratings - user_biases_n - item_biases[n])),axis=0)
        
        item_vec[n] = np.linalg.inv(u_n) @ ur_n

        

    return ([user_biases, item_biases], [user_vec, item_vec])

def alt_least_sqrt_with_feature(train,test,biases, latent_vec,feature ,lam = 2e-3, gamma = 3e-2, tau=1e-3,k=5):
    U,V = train
    M, N = len(U), len(V)
    
    user_biases, item_biases = biases
    user_vec, item_vec = latent_vec
    feature_vec, feat_hot = feature

    for m in range(M):
        if len(U[m]) > 0:
        # Calculate user bias
            user_ratings = np.array([r for (_, r) in U[m]])
            item_indices = np.array([n for (n, _) in U[m]])
            
            item_biases_m = item_biases[item_indices]
            item_count = len(item_indices)
            user_biases[m] = np.sum(lam * (user_ratings - np.dot(user_vec[m], item_vec[item_indices].T)- item_biases_m)) / ((lam * item_count) + gamma)
        
        # Compute item vector update
        v_n = np.zeros((k, k))
        vr_n = np.zeros(k)

        v_n = np.sum(np.einsum("ij,il->ijl",item_vec[item_indices],item_vec[item_indices]),axis=0) + tau * np.eye(k)
        vr_n = np.sum(np.einsum("ji,j->ji",item_vec[item_indices],(user_ratings - user_biases[m] -item_biases_m)), axis=0) 

        user_vec[m] = np.linalg.inv(v_n) @ vr_n


    for n in range(N):
        # Calculate item bias
        if len(V[n]) > 0:
            item_ratings = np.array([r for (_, r) in V[n]])
            user_indices = np.array([m for (m, _) in V[n]])


            user_biases_n = user_biases[user_indices]
            user_count = len(user_indices)
            item_biases[n] = np.sum(lam * (item_ratings - np.dot(user_vec[user_indices], item_vec[n].T) - user_biases_n)) / ((lam * user_count) + gamma)
        
        #feature
        n_feat_indices = np.where(np.array(feat_hot[n]) == 1)[0]
        n_feature_vec = tau/np.sqrt(len(n_feat_indices)) * np.sum(feature_vec[np.array(n_feat_indices)],axis=0)
        
        # Compute item vector update
        u_n =np.sum(np.einsum("ij,il->ijl",user_vec[user_indices],user_vec[user_indices]),axis=0) + tau * np.eye(k)
        ur_n = np.sum(np.einsum("ji,j->ji",user_vec[user_indices],(item_ratings - user_biases_n - item_biases[n])),axis=0) - n_feature_vec
        
        item_vec[n] = np.linalg.inv(u_n) @ ur_n

    for i in range(len(feature_vec)):
        # get feature embeddings for other features except ith feature
        indices = np.where(np.array(feat_hot)[:, i] == 1) 
        feature_sums = np.sum(np.array(feat_hot)[indices], axis=1) 
        features_except_i = np.delete(np.array(feat_hot), i, axis=1)[indices] 
        f_except_i = np.delete(feature_vec, i, axis=0) 
        
        # calculate feature update
        total_right = np.sum(np.einsum('ij, i -> ij', item_vec[indices], 1/np.sqrt(feature_sums)) 
                             - np.einsum('i, ij-> ij', 1/feature_sums , np.einsum('ij, jk -> ik', features_except_i, f_except_i)), axis=0) # calculate feature update
        
        feature_vec[i] = total_right/(np.sum(1/np.sqrt(feature_sums)) - 1)

    return ([user_biases, item_biases], [user_vec, item_vec],feature_vec)

def residues(U,biases,latent_vec)->Tuple[float,float]:
    [user_biases, item_biases] = biases
    [user_vec, item_vec] = latent_vec

    # calculate the sum and residue
    residue_sum = 0
    count = 0

    for m,V in enumerate(U):
        user_ratings = np.array([r for (_, r) in U[m]])
        item_indices = np.array([n for (n, _) in U[m]])
        item_biases_m = item_biases[item_indices.astype(int)]
        item_count = len(item_indices)
        # Calculate predictions for user m
        predictions = np.dot(user_vec[m], item_vec[item_indices.astype(int)].T) + item_biases_m + user_biases[m]
        
        # Calculate residuals
        residuals = user_ratings - predictions
        residue_sum += np.sum(residuals ** 2)
        count += item_count

    return residue_sum , count

def loss_func(u,biases,latent_vec,lam,gamma,tau)->float:
    [user_biases, item_biases] = biases
    [user_vec, item_vec] = latent_vec
    residue , _ = residues(u,biases,latent_vec)

    return (0.5 * lam * residue )- \
            0.5*gamma * (np.sum(user_biases) + np.sum(item_biases)) - \
                0.5 * tau * (np.einsum('ij,ij->', user_vec, user_vec) + np.einsum('ij,ij->', item_vec, item_vec))

def feat_loss_func(u,biases,latent_vec,feature,lam,gamma,tau)->float:
    [user_biases, item_biases] = biases
    [user_vec, item_vec] = latent_vec
    residue , _ = residues(u,biases,latent_vec)
    feature_vec, feat_hot = feature
    n_feat_vec = np.einsum("ij,ki->jk",feature_vec,feat_hot)
    item_feat_vec = item_vec - n_feat_vec.T

    return (0.5 * lam * residue )- \
            0.5*gamma * (np.sum(user_biases) + np.sum(item_biases)) - \
                0.5 * tau * (np.einsum('ij,ij->', user_vec, user_vec) + np.einsum('ij,ij->', item_feat_vec, item_feat_vec) +\
                    np.einsum('ij,ij->', feature_vec,feature_vec) )

def rmse (U,biases,latent_vec)->float:
    residue, count = residues(U,biases,latent_vec)
    return np.sqrt(residue /count)

def plot_fig(x,y,label,title,axlable,figsize=(6, 6), fontsize=20,data2=None):
    fig, ax = plt.subplots(figsize=figsize)

    if data2 is not None:
        ax.plot(x,data2,label=axlable[1])

    ax.plot(x,y,color='g',label=axlable[0])
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax.set_title(title, fontsize=fontsize, pad=20) 
    plt.legend()      
    plt.show()

def train(train,test,N,lam = 2e-3, gamma = 3e-2, tau=1e-3,k=300):
    """
    Train the bias for user and items

    Args:
        train (npdarray): training rating array
        tes (npdarray): testing rating array
        N (int): Number of Epoch
        lam (float): lambda parameter  
        gamma (float): gamma parameter
        tau (float) : tau parameter
        k (int): size of latent vector

    Returns:
        Tuple : loss_array, rmse_array , user_bia,item_bia 

    """
    #initialization
    U,V = train
    U_test,_ = test
    user_biases = np.zeros((len(U)))
    item_biases = np.zeros((len(V)))
    
    user_vec = np.random.normal(0,(1/np.sqrt(k)),(len(U),k))
    item_vec = np.random.normal(0,(1/np.sqrt(k)),(len(V),k))
    loss = np.zeros((N))
    rmse_ar = np.zeros((N))
    test_rmse_ar = np.zeros((N))

    #function parameters
    biases = [ user_biases, item_biases ]
    latent_vec = [user_vec, item_vec]

    

    for i in tqdm(range(N), desc="Epoch"):
        #alternate least squares
    #    biases , latent_vec =  alt_least_sqrt_biaOnly(train,test,biases,latent_vec,lam , gamma, tau,k)
       biases , latent_vec =  alt_least_sqrt(train,test,biases,latent_vec,lam , gamma, tau,k)
       loss[i] = loss_func(U, biases,latent_vec, lam, gamma,tau)


    
       rmse_ar[i] = rmse (U, biases,latent_vec)
       test_rmse_ar[i] = rmse(U_test,biases,latent_vec)
       print(f"train loss: {loss[i]}    train rmse: {rmse_ar[i]}    test rmse: {test_rmse_ar[i]}")
    
    data = {
        'lambda':lam,
        'gamma':gamma,
        'tau':tau,
        'k':k,
        'user_biases' : biases[0],
        'item_biases' : biases[1],
        'user_vec' : latent_vec[0],
        'item_vec' : latent_vec[1],
        'loss' : loss,
        'rmse_arr' : rmse_ar,
        'rmse_test' : test_rmse_ar
    }
   
    save_model(data,f"mdl_ltkg_{k}")

    return (loss,rmse_ar,test_rmse_ar, user_biases, item_biases)

def train_with_feature_vec(train,test,feature,N,lam = 2e-3, gamma = 3e-2, tau=1e-3,k=300):
    """
    Train the bias for user and items

    Args:
        train (npdarray): training rating array
        test (npdarray): testing rating array
        feature (list): feature list and one-hot of each feature
        N (int): Number of Epoch
        lam (float): lambda parameter  
        gamma (float): gamma parameter
        tau (float) : tau parameter
        k (int): size of latent vector

    Returns:
        Tuple : loss_array, rmse_array , user_bia,item_bia 

    """
    #initialization
    U,V = train
    U_test,_ = test
    user_biases = np.zeros((len(U)))
    item_biases = np.zeros((len(V)))
    n_feature = len(feature[0])

    user_vec = np.random.normal(0,(1/np.sqrt(k)),(len(U),k))
    item_vec = np.random.normal(0,(1/np.sqrt(k)),(len(V),k))
    feature_vec = np.random.normal(0,(1/0.9),(n_feature,k))
    loss = np.zeros((N))
    rmse_ar = np.zeros((N))
    test_rmse_ar = np.zeros((N))

    #function parameters
    biases = [ user_biases, item_biases ]
    latent_vec = [user_vec, item_vec]

    for i in tqdm(range(N), desc="Epoch"):
        #alternate least squares
       biases , latent_vec,feature_vec =  alt_least_sqrt_with_feature(train,test,biases,latent_vec ,[feature_vec,feature[1]],lam , gamma, tau,k)
       loss[i] = feat_loss_func(U, biases,latent_vec,[feature_vec,feature[1]], lam, gamma,tau)
    
       rmse_ar[i] = rmse (U, biases,latent_vec)
       test_rmse_ar[i] = rmse(U_test,biases,latent_vec)
       print(f"train loss: {loss[i]}    train rmse: {rmse_ar[i]}    test rmse: {test_rmse_ar[i]}")
    
    data = {
        'lambda':lam,
        'gamma':gamma,
        'tau':tau,
        'k':k,
        'user_biases' : biases[0],
        'item_biases' : biases[1],
        'user_vec' : latent_vec[0],
        'item_vec' : latent_vec[1],
        'feature_vec': feature_vec,
        'loss' : loss,
        'rmse_arr' : rmse_ar,
        'rmse_test' : test_rmse_ar
    }
   
    save_model(data,f"feat_mdl_ltkg_{k}")

    return (loss,rmse_ar,test_rmse_ar, user_biases, item_biases)

def run_parallel(thread_batch):
    """
    param:
    thread_batch: a list of tuple

    return:
    decorated function
    The is create threads according the number of thread_batch
    """

    def wrapper(func):
        def exec(*args, **kwargs):
            # Create processes to modify the nested list
            with  concurrent.futures.ProcessPoolExecutor() as exer:
                results = exer.map(func,thread_batch)
                result = [result for result in results]

            print(result)           
        return exec

    return wrapper