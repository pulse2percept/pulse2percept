import unittest
import os
from datasets import fetch_url

class TestBase(unittest.TestCase):
    # remove files after unittest is completed
    def tearDown(self):
        os.remove('../data/paper1.pdf')
        os.remove('../data/paper2.pdf')
        os.remove('../data/paper3.pdf')
        
    def test_fetch_url(self):
        url1 = 'https://www.nature.com/articles/s41598-019-45416-4.pdf'
        file_path1 = '../data/paper1.pdf'
        paper_checksum1 = 'e8a2db25916cdd15a4b7be75081ef3e57328fa5f335fb4664d1fb7090dcd6842'
        fetch_url(url1, file_path1, paper_checksum1)
        self.assertTrue(os.path.exists(file_path1))
        
        url2 = 'https://bionicvisionlab.org/publication/2019-optimal-surgical-placement/2019-optimal-surgical-placement.pdf'
        file_path2 = '../data/paper2.pdf'
        paper_checksum2 = 'e2d0cbecc9c2826f66f60576b44fe18ad6a635d394ae02c3f528b89cffcd9450'
        fetch_url(url2, file_path2, paper_checksum2)
        self.assertTrue(os.path.exists(file_path2))
        
        url3 = 'https://bionicvisionlab.org/publication/2017-pulse2percept/2017-pulse2percept.pdf'
        file_path3 = '../data/paper3.pdf'
        # Use wrong sha256 checksum of the file to test the result
        self.assertRaises(IOError, fetch_url, url3, file_path3, paper_checksum2)

if __name__ == '__main__':
    unittest.main()