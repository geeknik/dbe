import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import curriculum_learning_setup

class TestCurriculumLearning(unittest.TestCase):

    def test_sorts_by_last_octet(self):
        ips = ['192.168.1.3', '192.168.1.1', '192.168.1.2']
        sorted_ips = curriculum_learning_setup(ips)
        self.assertEqual(sorted_ips, ['192.168.1.1', '192.168.1.2', '192.168.1.3'])

    def test_filters_invalid_ips(self):
        ips = ['192.168.1.1', 'invalid', '192.168.1.2']
        sorted_ips = curriculum_learning_setup(ips)
        self.assertEqual(len(sorted_ips), 2)
        self.assertIn('192.168.1.1', sorted_ips)
        self.assertIn('192.168.1.2', sorted_ips)

    def test_exits_with_no_valid_ips(self):
        ips = ['invalid', 'not.an.ip']
        with self.assertRaises(SystemExit):
            curriculum_learning_setup(ips)

    def test_handles_ipv6(self):
        ips = ['::1', '2001:db8::1']
        sorted_ips = curriculum_learning_setup(ips)
        self.assertEqual(len(sorted_ips), 2)

if __name__ == '__main__':
    unittest.main()
