import unittest
import sys
import tempfile
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import Config

class TestConfig(unittest.TestCase):

    def test_load_defaults_when_file_missing(self):
        config = Config(config_path='/nonexistent/config.yaml')
        self.assertEqual(config.get('learning', 'max_episodes'), 1000)
        self.assertEqual(config.get('rewards', 'success'), 10)

    def test_load_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'learning': {'max_episodes': 500},
                'rewards': {'success': 20}
            }, f)
            config_path = f.name

        config = Config(config_path=config_path)
        self.assertEqual(config.get('learning', 'max_episodes'), 500)
        self.assertEqual(config.get('rewards', 'success'), 20)

        Path(config_path).unlink()

    def test_get_nested_value(self):
        config = Config(config_path='/nonexistent/config.yaml')
        value = config.get('learning', 'max_episodes')
        self.assertEqual(value, 1000)

    def test_get_with_default(self):
        config = Config(config_path='/nonexistent/config.yaml')
        value = config.get('nonexistent', 'key', default=42)
        self.assertEqual(value, 42)

    def test_get_deep_nested(self):
        config = Config(config_path='/nonexistent/config.yaml')
        value = config.get('network', 'remote_server')
        self.assertEqual(value, 'example.com')

if __name__ == '__main__':
    unittest.main()
