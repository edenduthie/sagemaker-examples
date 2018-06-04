import unittest
import prepare_data
import numpy as np
import shutil
import os

class TestPrepareData(unittest.TestCase):
    
    train_in_dir = '/Users/duthiee/data/gluon_recommender/train_in'
    train_out_dir = '/Users/duthiee/data/gluon_recommender/train_out'
    test_out_dir = '/Users/duthiee/data/gluon_recommender/test_out'

#    def test_load_df(self):
#        df = prepare_data.load_df(self.train_in_dir,nrows=10000)
#        self.assertEqual(len(df),17)
#        self.assertEqual(df.columns[0],'customer_id')
#        self.assertEqual(df.columns[1],'product_id')
#        self.assertEqual(df.columns[2],'star_rating')
#        self.assertEqual(df.columns[3],'user')
#        self.assertEqual(df.columns[4],'item')
#        
#    def test_split_n_fold(self):
#        X = np.array([0,1,2,3,4,5,6,7,8])
#        splits = prepare_data.split_n_fold(X,n_splits=2)
#        self.assertEqual(len(splits[0]),4)
#        self.assertEqual(len(splits[1]),4)
#        self.assertEqual(len(splits[2]),1)
#        
#        self.assertEqual(len(np.setdiff1d(splits[0],splits[1])),4)
#        self.assertEqual(len(np.setdiff1d(splits[0],splits[2])),4)
#        self.assertEqual(len(np.setdiff1d(splits[1],splits[2])),4)
#        
#    
#    def test_split_df(self):
#        df = prepare_data.load_df(self.train_in_dir,nrows=10000)
#        partitions = prepare_data.split_df(df,n_splits=2)
#        self.assertEqual(len(partitions),3)
#        total = 0
#        for partition in partitions:
#            total += len(partition)
#        self.assertEqual(total,len(df))
        
    def test_prepare_data(self):
        prepare_data.prepare_data(self.train_in_dir, 
                                  self.train_out_dir,
                                  self.test_out_dir,
                                  n_splits=2,nrows=10000)
        
        self.assertTrue(os.path.isfile(os.path.join(self.train_out_dir,'0_videos_train.csv')))
        self.assertTrue(os.path.isfile(os.path.join(self.train_out_dir,'1_videos_train.csv')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_out_dir,'videos_test.csv')))
        
        self.clear_dir(self.train_out_dir)
        self.clear_dir(self.test_out_dir)
        
    def clear_dir(self,path):
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        

if __name__ == '__main__':
    unittest.main()

