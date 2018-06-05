import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import boto3
import shutil

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
        
def upload_to_s3(source_dir,bucket,prefix):
    f = os.listdir(source_dir)
    client = boto3.client('s3')
    for filename in f:
        head,tail=os.path.split(filename)
        key = '{}/{}'.format(prefix,tail)
        client.upload_file(os.path.join(source_dir,filename), bucket, key)
        
def full_process(train_in_dir,train_out_dir,test_out_dir,bucket,train_prefix,text_prefix,
    n_splits=5,nrows=None):
    
    prepare_data(train_in_dir,train_out_dir,test_out_dir,n_splits,nrows)
    upload_to_s3(train_out_dir,bucket,train_prefix)
    upload_to_s3(test_out_dir,bucket,test_prefix)
    
def clear_dir(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    
if __name__ == '__main__':
    
    bucket = 'eduthie-sagemaker-1'
    prefix = 'gluon_recommender'
    train_in_dir = '/Users/duthiee/data/gluon_recommender/train_in'
    train_out_dir = '/Users/duthiee/data/gluon_recommender/train_out'
    test_out_dir = '/Users/duthiee/data/gluon_recommender/test_out'
    
    train_prefix = '{}/{}'.format(prefix,'train_dist')
    test_prefix = '{}/{}'.format(prefix,'test_dist')
    
    full_process(train_in_dir,train_out_dir,test_out_dir,bucket,train_prefix,test_prefix,
        n_splits=5)
    
    clear_dir(train_out_dir)
    clear_dir(test_out_dir)
        
        
