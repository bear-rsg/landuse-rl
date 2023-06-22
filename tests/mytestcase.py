import unittest


class MyTestCase(unittest.TestCase):
    def expectEqual(self, first, second, msg=None):
        with self.subTest():
            self.assertEqual(first, second, msg)

    def expectSequenceEqual(self, first, second, msg=None):
        with self.subTest():
            self.assertSequenceEqual(first, second, msg)

    def expectAlmostEqual(self, first, second, places=7, msg=None, delta=None):
        with self.subTest():
            self.assertAlmostEqual(first, second, places, msg, delta)

    def expectTrue(self, expr, msg=None):
        with self.subTest():
            self.assertTrue(expr, msg)

    def expectFalse(self, expr, msg=None):
        with self.subTest():
            self.assertFalse(expr, msg)

    def expectRaises(self, exception, callable, *args, **kwds):
        with self.subTest():
            self.assertRaises(exception, callable, *args, **kwds)
