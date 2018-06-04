import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def load_df(training_dir,nrows=None):
    f = os.listdir(training_dir)
    df = pd.read_csv(os.path.join(training_dir, f[0]), delimiter='\t', error_bad_lines=False, nrows=nrows)
    df = df[['customer_id', 'product_id', 'star_rating']]
    customers = df['customer_id'].value_counts()
    products = df['product_id'].value_counts()
    
    # Filter long-tail
    customers = customers[customers >= 5]
    products = products[products >= 10]

    reduced_df = df.merge(pd.DataFrame({'customer_id': customers.index})).merge(pd.DataFrame({'product_id': products.index}))
    customers = reduced_df['customer_id'].value_counts()
    products = reduced_df['product_id'].value_counts()

    # Number users and items
    customer_index = pd.DataFrame({'customer_id': customers.index, 'user': np.arange(customers.shape[0])})
    product_index = pd.DataFrame({'product_id': products.index, 'item': np.arange(products.shape[0])})

    reduced_df = reduced_df.merge(customer_index).merge(product_index)
    
    return reduced_df

# Splits the given array into n_splits parts randomly without replacement
# Each split is approximately the same size, except for perhaps the last one
def split_n_fold(X,n_splits=5):
    size = len(X)
    split_len = int(size/n_splits)
    outputs = []
    remaining = np.copy(X)
    for i in range(n_splits):
        split = np.random.choice(remaining,size=split_len,replace=False)
        outputs.append(split)
        remaining = np.setdiff1d(remaining,split)
    if len(remaining) > 0:
        outputs.append(remaining)
    return outputs

# Splits the given array into n_splits parts randomly without replacement by user id
# Each split is approximately the same size in terms of users, 
# except for perhaps the last one
def split_df(df,n_splits=5):
    users = df.user.unique()
    split_users = split_n_fold(users,n_splits)
    partitions = []
    for user_i in split_users:
        partition = df.loc[df['user'].isin(user_i)]
        partitions.append(partition)
    return partitions

def prepare_data(train_in_dir,train_out_dir,test_out_dir,n_splits=5,nrows=None):
    
    #load the data
    df = load_df(train_in_dir,nrows=nrows)
    
    # Split train and test
    test = df.groupby('customer_id').last().reset_index()
    train = df.merge(test[['customer_id', 'product_id']], on=['customer_id', 'product_id'], 
        how='outer', indicator=True)
    train = train[(train['_merge'] == 'left_only')]
    
    # write out test
    filename = os.path.join(test_out_dir,'videos_test.csv')
    test.to_csv(filename,index=False)
    
    # partition training data and write to file
    partitions = split_df(train,n_splits=n_splits)
    index = 0
    for partition in partitions:
        filename = os.path.join(train_out_dir,'{}_videos_train.csv'.format(index))
        partition.to_csv(filename,index=False)
        index += 1
        
        
