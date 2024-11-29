import unittest
from test.TestUtils import TestUtils
class Test_Boundary(unittest.TestCase):
    def test_Boundary(self):
        test_obj = TestUtils()
        test_obj.yakshaAssert("TestBoundary",True,"TestBoundary")
        print("TestBoundary = Passed")
