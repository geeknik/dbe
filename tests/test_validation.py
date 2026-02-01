import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import validate_ip, validate_url, Config

class TestValidation(unittest.TestCase):

    def test_validate_ip_valid(self):
        self.assertTrue(validate_ip('192.168.1.1'))
        self.assertTrue(validate_ip('10.0.0.1'))
        self.assertTrue(validate_ip('127.0.0.1'))
        self.assertTrue(validate_ip('::1'))
        self.assertTrue(validate_ip('2001:db8::1'))

    def test_validate_ip_invalid(self):
        self.assertFalse(validate_ip('256.256.256.256'))
        self.assertFalse(validate_ip('not.an.ip.address'))
        self.assertFalse(validate_ip('192.168.1'))
        self.assertFalse(validate_ip(''))
        self.assertFalse(validate_ip('192.168.1.1.1'))

    def test_validate_url_valid(self):
        self.assertTrue(validate_url('http://example.com'))
        self.assertTrue(validate_url('https://example.com'))
        self.assertTrue(validate_url('https://example.com/path/to/file'))
        self.assertTrue(validate_url('http://192.168.1.1:8080'))

    def test_validate_url_invalid_scheme(self):
        self.assertFalse(validate_url('ftp://example.com'))
        self.assertFalse(validate_url('file:///etc/passwd'))
        self.assertFalse(validate_url('javascript:alert(1)'))

    def test_validate_url_custom_schemes(self):
        self.assertTrue(validate_url('ftp://example.com', ['ftp']))
        self.assertFalse(validate_url('http://example.com', ['ftp']))

    def test_validate_url_malformed(self):
        self.assertFalse(validate_url('not a url'))
        self.assertFalse(validate_url(''))

if __name__ == '__main__':
    unittest.main()
