import unittest

from scenario_server.grading.graders import cosine_similarity 

class TestGraders(unittest.TestCase):

    def test_cosine_similarity(self):

        x = "apple"
        y = "pear"

        similarity_score = cosine_similarity(x,y)

        self.assertIsInstance(similarity_score,float)
        self.assertTrue(similarity_score < 1. )
        self.assertTrue(similarity_score > 0. )

        print(f"{similarity_score=}")

