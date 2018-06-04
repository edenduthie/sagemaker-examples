import unittest
import recommender

class TestRecommender(unittest.TestCase):
    
    train_dir = '/Users/duthiee/data/gluon_recommender/train'
    train_dir_ready = '/Users/duthiee/data/gluon_recommender/train_ready'

    def test_prepare_train_data(self):
        train_iter, test_iter, customer_index, product_index = recommender.prepare_train_data(self.train_dir,nrows=100000)
        self.assertEqual(len(train_iter),24)
        self.assertEqual(len(test_iter),12)
        self.assertEqual(len(customer_index),470)
        self.assertEqual(len(product_index),688)
        self.assertEqual(customer_index.columns[0],'customer_id')
        self.assertEqual(customer_index.columns[1],'user')
        self.assertEqual(product_index.columns[0],'item')
        self.assertEqual(product_index.columns[1],'product_id')

if __name__ == '__main__':
    unittest.main()

