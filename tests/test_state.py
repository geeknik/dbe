import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import State, update_state

class TestState(unittest.TestCase):

    def test_state_creation(self):
        state = State('192.168.1.1', 5)
        self.assertEqual(state.ip, '192.168.1.1')
        self.assertEqual(state.state_number, 5)

    def test_update_state_action_0(self):
        state = State('192.168.1.1', 5)
        new_state = update_state(state, 0)
        self.assertEqual(new_state.state_number, 6)
        self.assertEqual(new_state.ip, '192.168.1.1')

    def test_update_state_action_1(self):
        state = State('192.168.1.1', 5)
        new_state = update_state(state, 1)
        self.assertEqual(new_state.state_number, 4)

    def test_update_state_action_1_boundary(self):
        state = State('192.168.1.1', 0)
        new_state = update_state(state, 1)
        self.assertEqual(new_state.state_number, 0)

    def test_update_state_action_2(self):
        state = State('192.168.1.1', 5)
        new_state = update_state(state, 2)
        self.assertEqual(new_state.state_number, 10)

    def test_update_state_action_3(self):
        state = State('192.168.1.1', 10)
        new_state = update_state(state, 3)
        self.assertEqual(new_state.state_number, 5)

    def test_update_state_action_4(self):
        state = State('192.168.1.1', 5)
        new_state = update_state(state, 4)
        self.assertEqual(new_state.state_number, 10)

    def test_update_state_invalid_action(self):
        state = State('192.168.1.1', 5)
        new_state = update_state(state, 99)
        self.assertEqual(new_state, state)

if __name__ == '__main__':
    unittest.main()
