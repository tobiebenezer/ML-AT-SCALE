from Practical2 import *

if __name__ == "__main__":

    # load_data("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip","ml-latest-small.zip")
    # load_data("https://files.grouplens.org/datasets/movielens/ml-25m.zip",'ml-25m.zip')
    
    train_data, test_data, feature = get_data('ml-25m')
    # train_data, test_data , feature = get_data('ml-latest-small')
    print('>>>>>> data loaded >>>>>>>>>')

    train_data = [train_data['user_to_rating'],train_data['movie_to_rating']]
    test_data = [test_data['user_to_rating'],test_data['movie_to_rating']]
   
    #Parameter definition
    N=100
    lam=5
    gamma=2e-1
    tau = 0.4
    # lam=1e-3
    # gamma=1e-4
    # tau = 1e-4
    k = 5

    # loss_ar,rmse_ar,test_rmse_ar,_,_ = train(train_data,test_data,N ,lam ,gamma,tau,k)
    loss_ar,rmse_ar,test_rmse_ar,_,_ = train_with_feature_vec(train_data,test_data, feature,N ,lam ,gamma,tau,k)
    plot_fig(np.arange(N),loss_ar,['Iteration','Loss'],"Training Loss",['train'])
    plot_fig(np.arange(N),rmse_ar,['Iteration','Loss'],"Training RMSE",['train','test'],data2=test_rmse_ar)