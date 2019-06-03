import unittest
import variable as gates


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

class TestXORGate(unittest.TestCase):
    def testXORGate1(self):
        self.assertEqual(0, gates.XORGate(1,1))
    def testXORGate2(self):
        self.assertEqual(1, gates.XORGate(1,0))
    def testXORGate3(self):
        self.assertEqual(1, gates.XORGate(0,1))
    def testXORGate4(self):
        self.assertEqual(0, gates.XORGate(0,0))

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

if __name__ == '__main__':
    unittest.main()