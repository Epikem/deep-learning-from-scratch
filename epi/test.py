import unittest
import numpy as np
import variable as gates
import layers as Layers

@unittest.SkipTest
class TestGates(unittest.TestCase):
    def testANDGate(self):
        self.assertEqual(1, gates.ANDGate(1,1))
        self.assertEqual(0, gates.ANDGate(1,0))
        self.assertEqual(0, gates.ANDGate(0,1))
        self.assertEqual(0, gates.ANDGate(0,0))

    def testNANDGate(self):
        self.assertEqual(0, gates.NANDGate(1,1))
        self.assertEqual(1, gates.NANDGate(1,0))
        self.assertEqual(1, gates.NANDGate(0,1))
        self.assertEqual(1, gates.NANDGate(0,0))

    def testORGate(self):
        self.assertEqual(1, gates.ORGate(1,1))
        self.assertEqual(1, gates.ORGate(1,0))
        self.assertEqual(1, gates.ORGate(0,1))
        self.assertEqual(0, gates.ORGate(0,0))

@unittest.SkipTest
class TestXORGate(unittest.TestCase):
    def testXORGate1(self):
        self.assertEqual(0, gates.XORGate(1,1))
    def testXORGate2(self):
        self.assertEqual(1, gates.XORGate(1,0))
    def testXORGate3(self):
        self.assertEqual(1, gates.XORGate(0,1))
    def testXORGate4(self):
        self.assertEqual(0, gates.XORGate(0,0))

@unittest.SkipTest
class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

class VirtualVariable:
    def __init__(self, value=0):
        self.assign(value)

    def assign(self, value=0):
        self.value = value
        return value

    def get(self):
        return self.value

    def copy(self):
        return VirtualVariable(self.get())

    def hardcopy(self):
        return self

@unittest.SkipTest
class TestVariable(unittest.TestCase):

    def test_variable_copy(self):
        a = VirtualVariable(3)
        self.assertEqual(a.get(),3)
        b = a.copy()
        self.assertEqual(a.get(),b.get())
        a.assign(4)
        self.assertEqual(a.get(),4)
        self.assertNotEqual(a.get(),b.get())
        pass

    def test_variable_hardcopy(self):
        a = VirtualVariable(3)
        self.assertEqual(a.get(),3)
        b = a.hardcopy()
        self.assertEqual(a.get(),b.get())
        a.assign(4)
        self.assertEqual(a.get(),4)
        self.assertEqual(a.get(),b.get())
        pass

class TestNumpyArray(unittest.TestCase):
    def test_numpy_array_initialization_1d(self):
        # np.array, np.matrix.ndim
        arr = np.array([1,2,1,4])
        self.assertEqual(arr.ndim, 1)
        self.assertListEqual(list(arr), [1,2,1,4])

    def test_numpy_array_initialization_2d(self):
        # np.array, np.matrix.ndim, np.matrix.tolist
        arr = np.array([[2,1],[3,4]])
        self.assertEqual(arr.ndim, 2)
        self.assertListEqual(arr.tolist(), [[2,1],[3,4]])

    pass

class TestMulLayer(unittest.TestCase):

    def test_multiplication_layer(self):
        l1 = Layers.MulLayer()
        o = l1.forward(2,3)
        self.assertEqual(o,6)
        dx, dy = l1.backward(0.1)
        self.assertAlmostEqual(dx,0.3)
        self.assertAlmostEqual(dy,0.2)
        np.testing.assert_almost_equal([dx,dy],[0.3,0.2])

if __name__ == '__main__':
    unittest.main()