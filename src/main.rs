use std::collections::HashMap;
use rand::Rng;
use std::process::Command;
use std::net::TcpStream;
use clap::{App, Arg};
use std::sync::Mutex;
use lazy_static::lazy_static;
use itertools::Itertools;
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct State {
    ip: String,
    state_number: u32,
}

struct LongTermMemory {
    memory: Mutex<Vec<(State, i32, i32, State)>>,
    capacity: usize,
}

impl LongTermMemory {
    fn new(capacity: usize) -> Self {
        LongTermMemory {
            memory: Mutex::new(Vec::new()),
            capacity,
        }
    }

    fn store(&self, state: State, action: i32, reward: i32, next_state: State) {
        let mut memory = self.memory.lock().unwrap();
        memory.push((state, action, reward, next_state));
        if memory.len() > self.capacity {
            memory.remove(0);
        }
    }

    fn retrieve(&self, state: &State) -> Vec<(State, i32, i32, State)> {
        let memory = self.memory.lock().unwrap();
        memory.iter().filter(|(s, _, _, _)| s == state).cloned().collect()
    }

    fn best_action(&self, state: &State) -> Option<i32> {
        let experiences = self.retrieve(state);
        if experiences.is_empty() {
            return None;
        }

        let mut actions: HashMap<i32, Vec<i32>> = HashMap::new();
        for (_, a, r, _) in experiences {
            actions.entry(a).or_insert_with(Vec::new).push(r);
        }

        actions.iter()
            .max_by(|a, b| {
                let mean_a: f64 = a.1.iter().map(|&x| x as f64).sum::<f64>() / a.1.len() as f64;
                let mean_b: f64 = b.1.iter().map(|&x| x as f64).sum::<f64>() / b.1.len() as f64;
                mean_a.partial_cmp(&mean_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&k, _)| k)
    }
}

lazy_static! {
    static ref LONG_TERM_MEMORY: LongTermMemory = LongTermMemory::new(1000);
}

fn symbolic_reasoning(state: &State) -> Option<i32> {
    if state.state_number < 10 {
        Some(0)  // Try infecting when in early states
    } else if state.state_number < 50 {
        Some(2)  // Propagate when in intermediate states
    } else if state.state_number >= 50 {
        Some(1)  // Perform self-healing when in later states
    } else {
        None
    }
}

fn try_infect(ip: &str) -> bool {
    match TcpStream::connect(format!("{}:22", ip)) {
        Ok(_) => true,
        Err(_) => false,
    }
}

fn perform_self_healing(ip: &str) -> bool {
    println!("Performing self-healing on {}: Attempting to remove malicious files and restart services.", ip);

    // --- Simulate removing a malicious file ---
    let malicious_file = "/tmp/malicious_payload.sh"; // Example malicious file
    match Command::new("rm").arg("-f").arg(malicious_file).output() {
        Ok(output) => {
            if output.status.success() {
                println!("Successfully removed malicious file: {}", malicious_file);
            } else {
                eprintln!("Failed to remove malicious file {}: {}", malicious_file, String::from_utf8_lossy(&output.stderr));
            }
        },
        Err(e) => {
            eprintln!("Error executing rm command: {}", e);
        }
    }

    // --- Simulate restarting a critical service ---
    // This command is highly platform-dependent. Examples:
    // Linux (systemd): systemctl restart apache2
    // Linux (SysVinit): service apache2 restart
    // macOS: sudo launchctl stop com.apple.apached && sudo launchctl start com.apple.apached
    let service_name = "apache2"; // Example service to restart
    let restart_command = if cfg!(target_os = "linux") {
        "systemctl"
    } else if cfg!(target_os = "macos") {
        "launchctl"
    } else {
        ""
    };

    if !restart_command.is_empty() {
        let output = if restart_command == "systemctl" {
            Command::new(restart_command).arg("restart").arg(service_name).output()
        } else if restart_command == "launchctl" {
            // macOS launchctl requires stop and then start
            let _ = Command::new("sudo").arg("launchctl").arg("stop").arg(format!("com.apple.{}", service_name)).output();
            Command::new("sudo").arg("launchctl").arg("start").arg(format!("com.apple.{}", service_name)).output()
        } else {
            // Fallback or error for unsupported OS
            eprintln!("Unsupported OS for service restart command.");
            return false;
        };

        match output {
            Ok(output) => {
                if output.status.success() {
                    println!("Successfully restarted service: {}", service_name);
                    return true;
                } else {
                    eprintln!("Failed to restart service {}: {}", service_name, String::from_utf8_lossy(&output.stderr));
                }
            },
            Err(e) => {
                eprintln!("Error executing service restart command: {}", e);
            }
        }
    }

    false // Return false if any critical step fails or is not supported
}

fn propagate(ip_range: &[String]) -> bool {
    let mut vulnerable_machines = vec![];

    for ip in ip_range {
        if is_vulnerable(ip) {
            vulnerable_machines.push(ip.clone());
        }
    }

    let mut infected_any = false;
    for ip in &vulnerable_machines {
        if try_infect(ip) {
            println!("Successfully infected: {}", ip);
            infected_any = true;
        } else {
            println!("Failed to infect: {}", ip);
        }
    }
    infected_any
}

fn is_vulnerable(ip: &str) -> bool {
    // This is a basic vulnerability check: it attempts to connect to common ports.
    // A more sophisticated check would involve banner grabbing, exploit attempts, etc.
    let common_ports = [21, 22, 23, 80, 443, 3389]; // FTP, SSH, Telnet, HTTP, HTTPS, RDP

    for port in common_ports.iter() {
        match TcpStream::connect(format!("{}:{}", ip, port)) {
            Ok(_) => {
                println!("Port {} is open on {}. Considering vulnerable.", port, ip);
                return true; // Found an open port, consider it vulnerable for this example
            },
            Err(_) => {
                // Port is closed or unreachable, continue to next port
            }
        }
    }
    false // No common ports found open
}

fn check_self_awareness(_ip: &str) -> bool {
    // This implementation uses `ps` command to get memory usage.
    // It might be platform-dependent (Linux/macOS compatible).
    // For Windows, a different approach would be needed.
    let pid = std::process::id();
    let output = Command::new("ps")
        .arg("-p")
        .arg(pid.to_string())
        .arg("-o")
        .arg("rss=") // Resident Set Size in kilobytes
        .output();

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            if let Some(mem_kb_str) = stdout.trim().split('\n').next() {
                if let Ok(mem_kb) = mem_kb_str.parse::<u64>() {
                    // Define a threshold for memory usage (e.g., 100 MB)
                    let max_memory_mb = 100;
                    let mem_mb = mem_kb / 1024;
                    println!("Current memory usage: {} MB", mem_mb);
                    return mem_mb < max_memory_mb;
                }
            }
            eprintln!("Failed to parse memory usage from ps output.");
            false
        },
        Err(e) => {
            eprintln!("Failed to execute ps command for self-awareness check: {}", e);
            false
        }
    }
}


fn explore_environment(_ip: &str) -> bool {
    // This implementation uses `hostname` and `ip` commands.
    // It might be platform-dependent (Linux/macOS compatible).
    // For Windows, a different approach would be needed.
    println!("Exploring environment...");

    // Get hostname
    match Command::new("hostname").output() {
        Ok(output) => {
            let hostname = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("Hostname: {}", hostname);
        },
        Err(e) => eprintln!("Failed to get hostname: {}", e),
    }

    // Get IP addresses (using `ip addr show` or `ifconfig`)
    let ip_command = if cfg!(target_os = "linux") {
        "ip"
    } else if cfg!(target_os = "macos") {
        "ifconfig"
    } else {
        ""
    };

    if !ip_command.is_empty() {
        let output = if ip_command == "ip" {
            Command::new(ip_command).arg("addr").arg("show").output()
        } else {
            Command::new(ip_command).output()
        };

        match output {
            Ok(output) => {
                let ip_info = String::from_utf8_lossy(&output.stdout).trim().to_string();
                println!("IP Information:\n{}", ip_info);
            },
            Err(e) => eprintln!("Failed to get IP information: {}", e),
        }
    }

    true
}

fn update_state(state: &State, action: i32) -> State {
    match action {
        0 => State { ip: state.ip.clone(), state_number: state.state_number + 1 },
        1 => State { ip: state.ip.clone(), state_number: state.state_number.saturating_sub(1) },
        2 => State { ip: state.ip.clone(), state_number: state.state_number * 2 },
        3 => State { ip: state.ip.clone(), state_number: state.state_number / 2 },
        4 => State { ip: state.ip.clone(), state_number: state.state_number + 5 },
        _ => state.clone(),
    }
}



fn curriculum_learning_setup(ip_range: Vec<String>) -> Vec<String> {
    ip_range.into_iter().sorted_by_key(|ip| {
        ip.split('.').last().and_then(|octet| octet.parse::<u32>().ok()).unwrap_or(u32::MAX)
    }).collect()
}

fn choose_action(q_table: &HashMap<State, HashMap<i32, f64>>, state: &State, exploration_probability: f64) -> i32 {
    if rand::random::<f64>() < exploration_probability {
        return rand::thread_rng().gen_range(0..5); // Explore: Choose a random action
    }

    if let Some(best_ltm_action) = LONG_TERM_MEMORY.best_action(state) {
        return best_ltm_action;
    }

    if let Some(symbolic_action) = symbolic_reasoning(state) {
        return symbolic_action;
    }

    if let Some(actions) = q_table.get(state) {
        if let Some((&action, _)) = actions.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)) {
            return action;
        }
    }
    // Fallback to a random action if no better option is found
    rand::thread_rng().gen_range(0..5)
}

fn take_action(action: i32, current_state: &State) -> (i32, State) {
    let ip = &current_state.ip;
    let reward = match action {
        0 => if try_infect(ip) { 10 } else { -5 },
        1 => if perform_self_healing(ip) { 10 } else { -5 },
        2 => if propagate(&[ip.to_string()]) { 10 } else { -5 },
        3 => if check_self_awareness(ip) { 10 } else { -5 },
        4 => if explore_environment(ip) { 10 } else { -5 },
        _ => -5,
    };
    let next_state = update_state(current_state, action);
    (reward, next_state)
}

fn main() {
    let matches = App::new("Self-Improving Agent")
        .arg(Arg::with_name("ip_range")
            .required(true)
            .multiple(true)
            .help("List of IP addresses to scan"))
        .get_matches();

    let ip_range: Vec<String> = matches.values_of("ip_range").unwrap().map(String::from).collect();

    let mut q_table: HashMap<State, HashMap<i32, f64>> = HashMap::new();
    let mut exploration_probability = 1.0;
    let exploration_decay = 0.995;
    let max_episodes = 1000;
    let max_state = 10000;

    let ip_range = curriculum_learning_setup(ip_range);

    for episode in 0..max_episodes {
        for ip in &ip_range {
            let mut state = State { ip: ip.clone(), state_number: 0 };

            while state.state_number < max_state {
                if !q_table.contains_key(&state) {
                    q_table.insert(state.clone(), (0..5).map(|a| (a, 0.0)).collect());
                }

                let action = choose_action(&q_table, &state, exploration_probability);
                let (reward, next_state) = take_action(action, &state);

                if !q_table.contains_key(&next_state) {
                    q_table.insert(next_state.clone(), (0..5).map(|a| (a, 0.0)).collect());
                }

                let next_action = choose_action(&q_table, &next_state, exploration_probability);
                let q_value_next_state = {
                    let next_state_q_values = q_table.get(&next_state).unwrap();
                    *next_state_q_values.get(&next_action).unwrap()
                };
                let current_q = q_table.get_mut(&state).unwrap().get_mut(&action).unwrap();
                *current_q += 0.1 * (reward as f64 + 0.9 * q_value_next_state - *current_q);

                LONG_TERM_MEMORY.store(state.clone(), action, reward, next_state.clone());

                exploration_probability *= exploration_decay;
                state = next_state;
            }

            if episode % 10 == 0 {
                perform_self_healing(&state.ip);
                propagate(&ip_range);
            }
        }
    }
}
