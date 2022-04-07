import ps5
import pylab
import unittest 

class MyTest(unittest.TestCase):
    def test_evaluate_models_on_training(self):
        x = pylab.array([1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007])
        y = pylab.array([1.15, 1.19, 1.22, 1.21, 1.23, 1.27, 1.37, 1.30, 1.31])
        self.assertEqual(len(x), len(y), 'x, y arrays are different lengths')
        degs = [1, 5, 9]
        models = ps5.generate_models(x, y, degs)
        ps5.evaluate_models_on_training(x, y, models)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(MyTest))
    unittest.TextTestRunner(verbosity=2).run(suite)
