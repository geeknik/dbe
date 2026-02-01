import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import LongTermMemory, State

class TestLongTermMemory(unittest.TestCase):

    def setUp(self):
        self.memory = LongTermMemory(capacity=5)

    def test_store_and_retrieve(self):
        state1 = State('192.168.1.1', 0)
        state2 = State('192.168.1.2', 0)

        exp1 = (state1, 0, 10, State('192.168.1.1', 1))
        exp2 = (state1, 1, -5, State('192.168.1.1', 0))
        exp3 = (state2, 0, 10, State('192.168.1.2', 1))

        self.memory.store(exp1)
        self.memory.store(exp2)
        self.memory.store(exp3)

        retrieved = self.memory.retrieve(state1)
        self.assertEqual(len(retrieved), 2)
        self.assertIn(exp1, retrieved)
        self.assertIn(exp2, retrieved)

        retrieved2 = self.memory.retrieve(state2)
        self.assertEqual(len(retrieved2), 1)
        self.assertIn(exp3, retrieved2)

    def test_capacity_limit(self):
        for i in range(10):
            state = State(f'192.168.1.{i}', 0)
            exp = (state, 0, 10, State(f'192.168.1.{i}', 1))
            self.memory.store(exp)

        self.assertEqual(len(self.memory.memory), 5)

    def test_best_action(self):
        state = State('192.168.1.1', 0)

        self.memory.store((state, 0, 10, State('192.168.1.1', 1)))
        self.memory.store((state, 0, 8, State('192.168.1.1', 1)))
        self.memory.store((state, 1, -5, State('192.168.1.1', 0)))
        self.memory.store((state, 1, -3, State('192.168.1.1', 0)))

        best = self.memory.best_action(state)
        self.assertEqual(best, 0)

    def test_best_action_no_experience(self):
        state = State('192.168.1.1', 0)
        best = self.memory.best_action(state)
        self.assertIsNone(best)

if __name__ == '__main__':
    unittest.main()
