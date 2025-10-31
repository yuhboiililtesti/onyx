# ==============================
# 1. Imports & Environment Setup
# ==============================
import os
import sys
import json
import yaml
import shutil
import time
import threading
import multiprocessing
import logging
import logging.handlers
import atexit
import signal
import uuid
import tempfile
import subprocess
import hashlib
import random
import string
from pathlib import Path
from collections import deque
from typing import Dict, Any, List, Optional, Callable
import asyncio
import contextlib
import socket
import functools
import ctypes
import inspect
import platform

# Third-party packages
try:
    import psutil
    import prompt_toolkit
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
except ImportError as e:
    print(f"[ERROR] Missing dependencies: {e}")
    print("Install with: pip install psutil prompt_toolkit rich pyyaml")
    sys.exit(1)

console = Console()

# ==============================
# 2. Global Constants
# ==============================
ONYX_HOME = Path.home() / ".onyx"
CONFIG_FILE = ONYX_HOME / "config.yaml"
LOG_FILE = ONYX_HOME / "onyx.log"
SECRET_FILE = ONYX_HOME / "secret.json"

# Ensure directories exist
ONYX_HOME.mkdir(parents=True, exist_ok=True)

# ==============================
# 3. Configuration System
# ==============================
class ConfigManager:
    """Load, validate, and provide access to ONYX configuration."""
    defaults = {
        "execution_mode": "safe",
        "cli": {"prompt": "ONYX ▷ ", "history_file": str(ONYX_HOME / "cli_history.txt")},
        "logging": {"level": "INFO", "rotate": "midnight", "backup_count": 30},
        "supervisor": {"check_interval": 5},
        "network": {"monitor_interval": 10, "alert_threshold": 80},
        "cluster": {"enabled": False, "port": 5050, "token": None},
        "backup": {"directories": ["/etc", "/home"], "retention_days": 7},
        "ai": {"agents": 5, "epoch_interval": 10}
    }

    def __init__(self, config_path: Path = CONFIG_FILE):
        self.config_path = config_path
        self.data = {}
        self.load_config()

    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            self.data = self.defaults.copy()
            self.data.update(raw)
        else:
            self.data = self.defaults.copy()
            self.save_config()
        console.log(f"[CONFIG] Loaded configuration from {self.config_path}")

    def save_config(self):
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.data, f)
        console.log(f"[CONFIG] Saved configuration to {self.config_path}")

    def get(self, key, default=None):
        return self.data.get(key, default)

config_manager = ConfigManager()

# ==============================
# 4. Execution Mode
# ==============================
class ExecutionMode:
    """Handles safe vs real execution modes."""
    def __init__(self, mode: str = None):
        self.mode = mode or config_manager.get("execution_mode", "safe")
        self.mode = self.mode.lower()
        if self.mode not in ["safe", "real"]:
            console.log(f"[WARN] Unknown execution mode {self.mode}, defaulting to safe")
            self.mode = "safe"

    def execute(self, cmd: List[str], **kwargs):
        """Execute a system command based on mode."""
        if self.mode == "safe":
            console.log(f"[SAFE] Would execute: {' '.join(cmd)}")
            return 0, "", ""
        else:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
            out, err = proc.communicate()
            return proc.returncode, out.decode(), err.decode()

exec_mode = ExecutionMode()

# ==============================
# 5. Logging System
# ==============================
class LogManager:
    """JSON + rotating file logging system."""
    def __init__(self, log_file=LOG_FILE):
        self.logger = logging.getLogger("ONYX")
        self.logger.setLevel(getattr(logging, config_manager.get("logging")["level"].upper(), logging.INFO))
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
        )
        handler = logging.handlers.TimedRotatingFileHandler(
            str(log_file),
            when=config_manager.get("logging")["rotate"],
            backupCount=config_manager.get("logging")["backup_count"]
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)
        console.log(f"[INFO] {msg}")

    def warning(self, msg):
        self.logger.warning(msg)
        console.log(f"[WARN] {msg}")

    def error(self, msg):
        self.logger.error(msg)
        console.log(f"[ERROR] {msg}")

    def get_recent(self, n=20):
        """Read last n log lines."""
        try:
            with open(LOG_FILE, "r") as f:
                lines = deque(f, maxlen=n)
            return [line.strip() for line in lines]
        except Exception as e:
            return [f"[ERROR] Could not read logs: {e}"]

log_manager = LogManager()

# ==============================
# 6. Supervisor
# ==============================
class ThreadSupervisor:
    """Manage threads and graceful shutdown."""
    def __init__(self):
        self.threads: Dict[str, threading.Thread] = {}
        self.active_flags: Dict[str, threading.Event] = {}
        self.lock = threading.Lock()

    def register(self, name: str, target: Callable, daemon=False, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        stop_event = threading.Event()
        def wrapper(*args, **kwargs):
            while not stop_event.is_set():
                target(*args, **kwargs)
        thread = threading.Thread(target=wrapper, args=args, kwargs=kwargs, daemon=daemon, name=name)
        with self.lock:
            self.threads[name] = thread
            self.active_flags[name] = stop_event
        return thread

    def start_all(self):
        for name, thread in self.threads.items():
            thread.start()
            console.log(f"[SUPERVISOR] Started thread: {name}")

    def stop_all(self):
        for name, event in self.active_flags.items():
            event.set()
        for name, thread in self.threads.items():
            thread.join(timeout=5)
            console.log(f"[SUPERVISOR] Stopped thread: {name}")

supervisor = ThreadSupervisor()

# ==============================
# 7. System Façade
# ==============================
class SystemFacade:
    """Expose core APIs to CLI/TUI and services."""
    def __init__(self):
        self.config = config_manager
        self.logger = log_manager
        self.exec_mode = exec_mode
        self.supervisor = supervisor

    def show_system_status(self):
        status = {
            "uptime": time.time() - psutil.boot_time(),
            "cpu": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage("/")._asdict(),
        }
        return status

    def show_services_table(self):
        # Placeholder for service manager integration
        return [{"service": "ExampleService", "status": "running"}]

    def trigger_backup(self):
        # Placeholder for backup manager integration
        log_manager.info("Backup triggered via facade")

    def restart_system(self):
        log_manager.info("System restart requested")
        self.supervisor.stop_all()
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def show_ai_status(self):
        # Placeholder for AI metrics
        return {"epoch": 0, "last_loss": 0.0}

    def run_sandbox_test(self):
        # Placeholder for sandbox execution
        console.log("Running sandbox test...")
        return "Sandbox OK"

system_facade = SystemFacade()

# ==============================
# 8. CLI / TUI
# ==============================
class OnyxCommandCenter:
    """Interactive CLI using PromptToolkit and Rich."""
    def __init__(self, facade: SystemFacade):
        self.facade = facade
        self.session = PromptSession(history=prompt_toolkit.history.FileHistory(
            self.facade.config.get("cli")["history_file"]
        ))
        self.commands = {
            "help": self.help,
            "status": self.status,
            "exit": self.exit,
            "logs": self.show_logs,
            "backup": self.backup,
            "restart": self.restart,
            "sandbox": self.sandbox,
        }
        self.running = True

    def help(self, *args):
        console.print("Available commands:")
        for cmd in self.commands.keys():
            console.print(f"  - {cmd}")

    def status(self, *args):
        status = self.facade.show_system_status()
        console.print_json(json.dumps(status, indent=2))

    def show_logs(self, *args):
        logs = self.facade.logger.get_recent(20)
        for line in logs:
            console.print(line)

    def backup(self, *args):
        self.facade.trigger_backup()

    def restart(self, *args):
        self.facade.restart_system()

    def sandbox(self, *args):
        result = self.facade.run_sandbox_test()
        console.print(result)

    def exit(self, *args):
        console.print("[CLI] Exiting ONYX...")
        self.running = False
        self.facade.supervisor.stop_all()
        sys.exit(0)

    def start(self):
        console.print(Panel(Text("Welcome to ONYX CLI", style="bold cyan")))
        with patch_stdout():
            while self.running:
                try:
                    cmd_line = self.session.prompt(self.facade.config.get("cli")["prompt"])
                    if not cmd_line.strip():
                        continue
                    parts = cmd_line.strip().split()
                    cmd, *args = parts
                    func = self.commands.get(cmd)
                    if func:
                        func(*args)
                    else:
                        console.print(f"Unknown command: {cmd}")
                except (KeyboardInterrupt, EOFError):
                    self.exit()

onyx_cli = OnyxCommandCenter(system_facade)

# ==============================
# 9. Signal Handling & Graceful Shutdown
# ==============================
def shutdown(signum=None, frame=None):
    console.print(f"[SYSTEM] Received signal {signum}, shutting down...")
    supervisor.stop_all()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)
atexit.register(supervisor.stop_all)

# ==============================
# 10. Bootstrap / Start CLI
# ==============================
if __name__ == "__main__":
    log_manager.info(f"ONYX System initialized {{'node_id': '{uuid.uuid4().hex}'}}")
    log_manager.info("Starting ONYX CLI...")
    onyx_cli.start()

# ==============================
# Phase 2: AI & Multi-Agent System
# ==============================

import math
import pickle
from collections import defaultdict, deque
from threading import Lock, Event

# ==============================
# 1. Reward System
# ==============================
class RewardSystem:
    """Tracks points, rewards, and reinforcement feedback."""
    def __init__(self, storage_file: Path = ONYX_HOME / "reward_state.pkl"):
        self.storage_file = storage_file
        self._state = {"total": 0, "history": []}
        self.lock = Lock()
        self.load_state()

    def add_points(self, amount: int, reason: str):
        with self.lock:
            self._state["total"] += amount
            self._state["history"].append({"timestamp": time.time(), "amount": amount, "reason": reason})
            log_manager.info(f"[REWARD] {amount} points added for {reason}")
            self.save_state()

    def get_total(self) -> int:
        with self.lock:
            return self._state["total"]

    def get_history(self, last_n=10) -> List[Dict[str, Any]]:
        with self.lock:
            return self._state["history"][-last_n:]

    def save_state(self):
        try:
            with open(self.storage_file, "wb") as f:
                pickle.dump(self._state, f)
        except Exception as e:
            log_manager.error(f"[REWARD] Failed to save state: {e}")

    def load_state(self):
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "rb") as f:
                    self._state = pickle.load(f)
            except Exception as e:
                log_manager.error(f"[REWARD] Failed to load state: {e}")

reward_system = RewardSystem()

# ==============================
# 2. Decision Engine
# ==============================
class DecisionEngine:
    """Evaluates system metrics and selects top-priority goals."""
    def __init__(self, facade: SystemFacade):
        self.facade = facade

    def evaluate(self, context: Dict[str, Any]) -> str:
        """Return top goal based on metrics."""
        cpu = context.get("cpu", psutil.cpu_percent())
        memory = context.get("memory", psutil.virtual_memory().percent)
        last_action = context.get("last_action", None)

        # Heuristic scoring
        scores = {
            "optimization": max(0, 100 - cpu - memory),
            "backup": 50 if last_action != "backup" else 10,
            "idle_learning": 20
        }
        top_goal = max(scores, key=scores.get)
        log_manager.info(f"[DECISION] Evaluated context {context}, top_goal={top_goal}")
        return top_goal

decision_engine = DecisionEngine(system_facade)

# ==============================
# 3. Neural Core
# ==============================
import numpy as np

class NeuralCore:
    """In-memory neural network core for lightweight learning."""
    def __init__(self, input_size=5, hidden_size=16, output_size=3, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        # Random weight initialization
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        self.x = np.array(x)
        self.z1 = self.x @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.z2  # Linear output for now
        return self.a2

    def backward(self, grad_output):
        grad_z2 = grad_output
        grad_W2 = np.outer(self.a1, grad_z2)
        grad_b2 = grad_z2
        grad_a1 = self.W2 @ grad_z2
        grad_z1 = grad_a1 * (1 - self.a1 ** 2)
        grad_W1 = np.outer(self.x, grad_z1)
        grad_b1 = grad_z1

        # Update weights
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2

    def train(self, x, y_target):
        y_pred = self.forward(x)
        loss = np.mean((y_pred - y_target) ** 2)
        grad_output = 2 * (y_pred - y_target) / len(y_target)
        self.backward(grad_output)
        return loss

neural_core = NeuralCore()

# ==============================
# 4. AIAgent
# ==============================
class AIAgent:
    """Represents an autonomous agent performing tasks and learning."""
    def __init__(self, agent_id: int, facade: SystemFacade, exec_mode: ExecutionMode):
        self.agent_id = agent_id
        self.facade = facade
        self.exec_mode = exec_mode
        self.task_queue: deque = deque()
        self.active = True
        self.last_action: Optional[str] = None
        self.thread: Optional[threading.Thread] = None

    def add_task(self, task: Dict[str, Any]):
        self.task_queue.append(task)
        log_manager.info(f"[AGENT-{self.agent_id}] Task added: {task}")

    def process_task(self, task: Dict[str, Any]):
        task_type = task.get("type", "generic")
        if task_type == "backup":
            self.facade.trigger_backup()
            reward_system.add_points(10, "backup_completed")
        elif task_type == "optimization":
            console.log(f"[AGENT-{self.agent_id}] Optimizing resources...")
            reward_system.add_points(5, "optimization")
        elif task_type == "sandbox":
            result = self.facade.run_sandbox_test()
            console.log(f"[AGENT-{self.agent_id}] Sandbox result: {result}")
            reward_system.add_points(2, "sandbox")
        else:
            console.log(f"[AGENT-{self.agent_id}] Performing generic task")
            reward_system.add_points(1, "generic_task")

        self.last_action = task_type

    def idle_behavior(self):
        """Perform learning during idle time."""
        context = {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "last_action": self.last_action
        }
        top_goal = decision_engine.evaluate(context)
        self.last_action = top_goal
        if top_goal == "idle_learning":
            x = np.random.rand(neural_core.input_size)
            y_target = np.random.rand(neural_core.output_size)
            loss = neural_core.train(x, y_target)
            reward_system.add_points(int(1 / (1 + loss)), "idle_learning")
            log_manager.info(f"[AGENT-{self.agent_id}] Idle learning loss={loss}")

    def run_loop(self):
        while self.active:
            if self.task_queue:
                task = self.task_queue.popleft()
                self.process_task(task)
            else:
                self.idle_behavior()
            time.sleep(1)

    def start(self):
        self.thread = threading.Thread(target=self.run_loop, name=f"AIAgent-{self.agent_id}")
        self.thread.start()
        console.log(f"[AGENT-{self.agent_id}] Started")

    def terminate(self):
        self.active = False
        if self.thread:
            self.thread.join()
        console.log(f"[AGENT-{self.agent_id}] Terminated")

# ==============================
# 5. Multi-Agent Manager
# ==============================
class AIManager:
    """Manages multiple agents and schedules tasks."""
    def __init__(self, facade: SystemFacade, exec_mode: ExecutionMode, agent_count: int = 5):
        self.facade = facade
        self.exec_mode = exec_mode
        self.agent_count = agent_count
        self.agents: List[AIAgent] = []

    def initialize_agents(self):
        for i in range(self.agent_count):
            agent = AIAgent(i, self.facade, self.exec_mode)
            self.agents.append(agent)
            supervisor.register(f"AIAgent-{i}", agent.run_loop)
        console.log(f"[AI] Initialized {self.agent_count} agents")

    def start_agents(self):
        for agent in self.agents:
            agent.start()

    def stop_agents(self):
        for agent in self.agents:
            agent.terminate()

    def dispatch_task(self, task: Dict[str, Any]):
        """Round-robin dispatch."""
        agent = min(self.agents, key=lambda a: len(a.task_queue))
        agent.add_task(task)

ai_manager = AIManager(system_facade, exec_mode, agent_count=config_manager.get("ai")["agents"])
ai_manager.initialize_agents()
ai_manager.start_agents()

# ==============================
# 6. Analytics & Optimizer
# ==============================
class SystemOptimizer:
    """Monitor and adjust system resources."""
    def __init__(self, facade: SystemFacade):
        self.facade = facade

    def evaluate(self):
        status = self.facade.show_system_status()
        # Simple heuristic: if CPU > 85%, delay low-priority tasks
        high_cpu = status["cpu"] > 85
        if high_cpu:
            console.print("[OPTIMIZER] High CPU load detected, throttling idle tasks")
        return high_cpu

optimizer = SystemOptimizer(system_facade)

# ==============================
# 7. Integration Hooks
# ==============================
def ai_shutdown_hook():
    log_manager.info("[AI] Shutting down agents...")
    ai_manager.stop_agents()

atexit.register(ai_shutdown_hook)

# ==============================
# Phase 2 Bootstrapping
# ==============================
log_manager.info("[AI] Phase 2 AI & Multi-Agent System initialized")

# Example dispatch
ai_manager.dispatch_task({"type": "backup"})
ai_manager.dispatch_task({"type": "optimization"})
ai_manager.dispatch_task({"type": "sandbox"})

# ==============================
# 8. Persistent Agent Memory
# ==============================
class AIMemory:
    """Store agent experiences for reinforcement learning."""
    def __init__(self, storage_file: Path = ONYX_HOME / "ai_memory.pkl"):
        self.storage_file = storage_file
        self.data = defaultdict(list)
        self.lock = Lock()
        self.load_memory()

    def add_experience(self, agent_id: int, state: Dict[str, Any], action: str, reward: int, next_state: Dict[str, Any]):
        with self.lock:
            self.data[agent_id].append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "timestamp": time.time()
            })
            self.save_memory()

    def get_experiences(self, agent_id: int, last_n: int = 50):
        with self.lock:
            return self.data[agent_id][-last_n:]

    def save_memory(self):
        try:
            with open(self.storage_file, "wb") as f:
                pickle.dump(self.data, f)
        except Exception as e:
            log_manager.error(f"[AIMEMORY] Failed to save memory: {e}")

    def load_memory(self):
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "rb") as f:
                    self.data = pickle.load(f)
            except Exception as e:
                log_manager.error(f"[AIMEMORY] Failed to load memory: {e}")

ai_memory = AIMemory()

# ==============================
# 9. Advanced Task Scheduling
# ==============================
class TaskScheduler:
    """Priority-based task dispatcher for AI agents."""
    def __init__(self, ai_manager: AIManager):
        self.ai_manager = ai_manager
        self.task_queue: List[Tuple[int, Dict[str, Any]]] = []  # (priority, task)
        self.lock = Lock()
        self.active = True
        self.thread = threading.Thread(target=self.run_loop, name="TaskScheduler")
        self.thread.start()

    def add_task(self, task: Dict[str, Any], priority: int = 5):
        with self.lock:
            self.task_queue.append((priority, task))
            self.task_queue.sort(key=lambda x: x[0])  # Lower number = higher priority
            log_manager.info(f"[SCHEDULER] Task added: {task} with priority {priority}")

    def run_loop(self):
        while self.active:
            with self.lock:
                if self.task_queue:
                    _, task = self.task_queue.pop(0)
                    self.ai_manager.dispatch_task(task)
            time.sleep(0.5)

    def shutdown(self):
        self.active = False
        self.thread.join()
        log_manager.info("[SCHEDULER] Shutdown complete")

scheduler = TaskScheduler(ai_manager)

# ==============================
# 10. Reward Shaping Utilities
# ==============================
class RewardShaper:
    """Calculate rewards based on task complexity and system state."""
    def __init__(self, reward_system: RewardSystem):
        self.reward_system = reward_system

    def shape_reward(self, task_type: str, context: Dict[str, Any]) -> int:
        cpu = context.get("cpu", psutil.cpu_percent())
        memory = context.get("memory", psutil.virtual_memory().percent)
        base = 1
        if task_type == "backup":
            base = 10
        elif task_type == "optimization":
            base = 5
        elif task_type == "sandbox":
            base = 2
        # Penalize high-load actions
        if cpu > 85 or memory > 85:
            base = max(1, base // 2)
        self.reward_system.add_points(base, f"task_{task_type}")
        return base

reward_shaper = RewardShaper(reward_system)

# ==============================
# 11. Neural Exploration
# ==============================
class ExplorationModule:
    """Adds stochastic exploration to neural decisions."""
    def __init__(self, neural_core: NeuralCore, epsilon: float = 0.2):
        self.neural_core = neural_core
        self.epsilon = epsilon

    def decide(self, input_state: np.ndarray) -> int:
        """Return action index, sometimes random for exploration."""
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.neural_core.output_size)
            log_manager.debug(f"[EXPLORATION] Random action {action}")
        else:
            outputs = self.neural_core.forward(input_state)
            action = int(np.argmax(outputs))
            log_manager.debug(f"[EXPLORATION] Chosen action {action} from neural output {outputs}")
        return action

exploration_module = ExplorationModule(neural_core)

# ==============================
# 12. AI Agent Integration
# ==============================
for agent in ai_manager.agents:
    def agent_task_wrapper(agent=agent):
        while agent.active:
            experiences = ai_memory.get_experiences(agent.agent_id)
            if experiences:
                last_exp = experiences[-1]
                input_state = np.random.rand(neural_core.input_size)  # Placeholder
                action_index = exploration_module.decide(input_state)
                # Generate synthetic reward based on last action
                reward = reward_shaper.shape_reward(last_exp["action"], {"cpu": psutil.cpu_percent()})
                ai_memory.add_experience(agent.agent_id, last_exp["state"], last_exp["action"], reward, last_exp["next_state"])
            time.sleep(1)

    threading.Thread(target=agent_task_wrapper, name=f"AIAgentMemory-{agent.agent_id}").start()

# ==============================
# 13. Phase 2 Complete Bootstrap
# ==============================
log_manager.info("[AI] Phase 2 complete: Multi-Agent System fully operational")

# Example tasks
scheduler.add_task({"type": "backup"}, priority=1)
scheduler.add_task({"type": "optimization"}, priority=3)
scheduler.add_task({"type": "sandbox"}, priority=5)

# ==============================
# PHASE 3: Services, Networking & Cluster
# ==============================

# ==============================
# 1. Service Base Class
# ==============================
class Service:
    """Base class for all ONYX services."""
    def __init__(self, name: str, command: List[str], critical: bool = False):
        self.name = name
        self.command = command
        self.critical = critical
        self.process: Optional[subprocess.Popen] = None
        self.lock = Lock()
        self.active = False

    def start(self):
        with self.lock:
            if self.active:
                log_manager.info(f"[SERVICE] {self.name} already running")
                return
            try:
                log_manager.info(f"[SERVICE] Starting {self.name}")
                self.process = subprocess.Popen(self.command)
                self.active = True
            except Exception as e:
                log_manager.error(f"[SERVICE] Failed to start {self.name}: {e}")

    def stop(self):
        with self.lock:
            if self.process and self.active:
                self.process.terminate()
                self.process.wait()
                log_manager.info(f"[SERVICE] {self.name} stopped")
                self.active = False

    def health_check(self) -> bool:
        if not self.process or self.process.poll() is not None:
            log_manager.warning(f"[SERVICE] {self.name} unhealthy")
            return False
        return True


# ==============================
# 2. Docker Manager
# ==============================
class DockerManager(Service):
    """Manage Docker containers."""
    def __init__(self):
        super().__init__("DockerManager", ["docker", "info"], critical=True)
        import docker
        self.client = docker.from_env()

    def start_container(self, image: str, name: str, detach: bool = True, ports: Optional[Dict[int, int]] = None):
        try:
            container = self.client.containers.run(image, name=name, detach=detach, ports=ports)
            log_manager.info(f"[DOCKER] Started container {name}")
            return container
        except Exception as e:
            log_manager.error(f"[DOCKER] Failed to start container {name}: {e}")
            return None

    def stop_container(self, name: str):
        try:
            container = self.client.containers.get(name)
            container.stop()
            log_manager.info(f"[DOCKER] Stopped container {name}")
        except Exception as e:
            log_manager.error(f"[DOCKER] Failed to stop container {name}: {e}")


docker_manager = DockerManager()
docker_manager.start()


# ==============================
# 3. VM Manager
# ==============================
class VMManager(Service):
    """Manage virtual machines via libvirt."""
    def __init__(self):
        super().__init__("VMManager", ["virsh", "list"], critical=True)
        import libvirt
        self.conn = libvirt.open("qemu:///system")

    def start_vm(self, name: str):
        try:
            domain = self.conn.lookupByName(name)
            domain.create()
            log_manager.info(f"[VM] Started VM {name}")
        except Exception as e:
            log_manager.error(f"[VM] Failed to start VM {name}: {e}")

    def shutdown_vm(self, name: str):
        try:
            domain = self.conn.lookupByName(name)
            domain.shutdown()
            log_manager.info(f"[VM] Shutdown VM {name}")
        except Exception as e:
            log_manager.error(f"[VM] Failed to shutdown VM {name}: {e}")


vm_manager = VMManager()
vm_manager.start()


# ==============================
# 4. Plex / Media Manager
# ==============================
class PlexManager(Service):
    """Manage Plex Media Server."""
    def __init__(self):
        super().__init__("PlexManager", ["systemctl", "status", "plexmediaserver"])
        self.unit_name = "plexmediaserver"

    def restart(self):
        try:
            subprocess.run(["systemctl", "restart", self.unit_name], check=True)
            log_manager.info(f"[PLEX] Restarted Plex Media Server")
        except Exception as e:
            log_manager.error(f"[PLEX] Failed to restart Plex: {e}")

plex_manager = PlexManager()
plex_manager.restart()


# ==============================
# 5. Minecraft Manager
# ==============================
class MinecraftManager(Service):
    """Manage Minecraft servers."""
    def __init__(self, jar_url: str = "https://launcher.mojang.com/v1/objects/server.jar"):
        super().__init__("MinecraftManager", ["java", "-jar", "server.jar", "nogui"])
        self.jar_url = jar_url
        self.server_path = ONYX_HOME / "minecraft_server"
        self.server_path.mkdir(parents=True, exist_ok=True)
        self.download_jar()

    def download_jar(self):
        jar_file = self.server_path / "server.jar"
        if not jar_file.exists():
            try:
                import requests
                r = requests.get(self.jar_url, stream=True)
                with open(jar_file, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
                log_manager.info("[MINECRAFT] server.jar downloaded")
            except Exception as e:
                log_manager.error(f"[MINECRAFT] Failed to download server.jar: {e}")

minecraft_manager = MinecraftManager()
minecraft_manager.start()


# ==============================
# 6. VPN Manager
# ==============================
class VPNManager(Service):
    """Manage WireGuard VPN."""
    def __init__(self, conf_path: Path = Path("/etc/wireguard/onyx.conf")):
        super().__init__("VPNManager", ["wg-quick", "up", str(conf_path)])
        self.conf_path = conf_path

    def connect(self):
        try:
            subprocess.run(["wg-quick", "up", str(self.conf_path)], check=True)
            log_manager.info("[VPN] WireGuard connected")
        except Exception as e:
            log_manager.error(f"[VPN] WireGuard connection failed: {e}")

vpn_manager = VPNManager()
vpn_manager.connect()


# ==============================
# 7. Firewall Manager
# ==============================
class FirewallManager(Service):
    """Manage system firewall via nftables."""
    def __init__(self):
        super().__init__("FirewallManager", ["nft", "list", "ruleset"], critical=True)

    def apply_rule(self, rule: List[str]):
        import shlex
        try:
            subprocess.run(["nft"] + rule, check=True)
            log_manager.info(f"[FIREWALL] Applied rule: {rule}")
        except Exception as e:
            log_manager.error(f"[FIREWALL] Failed to apply rule {rule}: {e}")

firewall_manager = FirewallManager()


# ==============================
# 8. Cluster Node & Manager
# ==============================
class ClusterNode:
    """Represents a single ONYX cluster node."""
    def __init__(self, node_id: str = str(uuid.uuid4()), address: str = "127.0.0.1"):
        self.node_id = node_id
        self.address = address
        self.metrics: Dict[str, Any] = {}
        self.last_heartbeat = time.time()

    def update_metrics(self):
        self.metrics = {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage("/").percent,
            "uptime": time.time() - psutil.boot_time(),
        }
        self.last_heartbeat = time.time()

class ClusterManager:
    """Manage multi-node clusters and distributed agents."""
    def __init__(self):
        self.nodes: Dict[str, ClusterNode] = {}
        self.lock = Lock()
        self.active = True
        self.thread = threading.Thread(target=self.sync_loop, name="ClusterManager")
        self.thread.start()

    def add_node(self, node: ClusterNode):
        with self.lock:
            self.nodes[node.node_id] = node
            log_manager.info(f"[CLUSTER] Node {node.node_id} added")

    def remove_node(self, node_id: str):
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                log_manager.info(f"[CLUSTER] Node {node_id} removed")

    def sync_loop(self):
        while self.active:
            with self.lock:
                for node in self.nodes.values():
                    node.update_metrics()
            time.sleep(10)

    def shutdown(self):
        self.active = False
        self.thread.join()
        log_manager.info("[CLUSTER] ClusterManager shutdown complete")

cluster_manager = ClusterManager()


# ==============================
# 9. Network Monitor & Traffic Analyzer
# ==============================
class NetworkMonitor:
    """Monitor system network stats."""
    def __init__(self):
        self.prev_io = psutil.net_io_counters()
        self.lock = Lock()
        self.active = True
        self.thread = threading.Thread(target=self.monitor_loop, name="NetworkMonitor")
        self.thread.start()

    def monitor_loop(self):
        while self.active:
            io = psutil.net_io_counters()
            delta_sent = io.bytes_sent - self.prev_io.bytes_sent
            delta_recv = io.bytes_recv - self.prev_io.bytes_recv
            log_manager.debug(f"[NETWORK] Sent: {delta_sent}, Recv: {delta_recv}")
            self.prev_io = io
            time.sleep(1)

    def stop(self):
        self.active = False
        self.thread.join()
        log_manager.info("[NETWORK] NetworkMonitor stopped")

network_monitor = NetworkMonitor()


# ==============================
# 10. Phase 3 Bootstrap
# ==============================
log_manager.info("[SERVICES & CLUSTER] Phase 3 complete: Services, Networking & Cluster operational")

# Example Service Start
docker_manager.start_container("nginx:latest", "onyx_nginx", ports={80: 8080})
vm_manager.start_vm("onyx_test_vm")
plex_manager.restart()
minecraft_manager.start()
vpn_manager.connect()
firewall_manager.apply_rule(["add", "rule", "inet", "filter", "input", "tcp", "dport", "22", "accept"])
cluster_manager.add_node(ClusterNode(node_id="node1"))

# ==============================
# PHASE 4: Storage, Security & API/Web UI
# ==============================

# ==============================
# 1. Database & Persistence
# ==============================
class Database:
    """Simple SQLite-backed datastore for ONYX."""
    def __init__(self, db_path: Path = ONYX_HOME / "onyx.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.lock = Lock()
        self.init_tables()

    def init_tables(self):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS backups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                last_seen DATETIME,
                metrics TEXT
            )""")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS rewards (
                agent_id TEXT,
                points INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(agent_id)
            )""")
            self.conn.commit()

    def insert_backup(self, path: str, btype: str):
        with self.lock:
            self.conn.execute("INSERT INTO backups (path, type) VALUES (?, ?)", (path, btype))
            self.conn.commit()

    def get_backups(self, limit: int = 10):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM backups ORDER BY timestamp DESC LIMIT ?", (limit,))
            return cursor.fetchall()

db = Database()


# ==============================
# 2. Backup Manager
# ==============================
class BackupManager:
    """Manage system backups and rotations."""
    BACKUP_DIR = ONYX_HOME / "backups"

    def __init__(self):
        self.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()

    def create_backup(self, sources: List[Path], compress: bool = True):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}.tar.gz" if compress else f"backup_{timestamp}"
        backup_path = self.BACKUP_DIR / backup_name

        try:
            import tarfile
            mode = "w:gz" if compress else "w"
            with tarfile.open(backup_path, mode) as tar:
                for src in sources:
                    tar.add(src, arcname=src.name)
            log_manager.info(f"[BACKUP] Created backup {backup_path}")
            db.insert_backup(str(backup_path), "full")
            return backup_path
        except Exception as e:
            log_manager.error(f"[BACKUP] Failed to create backup: {e}")
            return None

    def rotate_backups(self, keep_last: int = 5):
        backups = sorted(self.BACKUP_DIR.glob("backup_*"), key=lambda f: f.stat().st_mtime, reverse=True)
        for old in backups[keep_last:]:
            try:
                old.unlink()
                log_manager.info(f"[BACKUP] Deleted old backup {old}")
            except Exception as e:
                log_manager.error(f"[BACKUP] Failed to delete backup {old}: {e}")

backup_manager = BackupManager()


# ==============================
# 3. Encryption / Keyring
# ==============================
class Keyring:
    """Simple symmetric key management."""
    def __init__(self, key_path: Path = ONYX_HOME / "key.bin"):
        self.key_path = key_path
        self.key: bytes = self.load_or_create_key()

    def load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            with open(self.key_path, "rb") as f:
                return f.read()
        else:
            key = os.urandom(32)
            with open(self.key_path, "wb") as f:
                f.write(key)
            return key

    def encrypt(self, data: bytes) -> bytes:
        from cryptography.fernet import Fernet
        f = Fernet(base64.urlsafe_b64encode(self.key))
        return f.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        from cryptography.fernet import Fernet
        f = Fernet(base64.urlsafe_b64encode(self.key))
        return f.decrypt(token)

keyring = Keyring()


# ==============================
# 4. Authentication
# ==============================
class AuthManager:
    """Simple password authentication."""
    SECRET_FILE = ONYX_HOME / "secret.json"

    def __init__(self):
        self.authenticated = False
        self.password_hash = self.load_password_hash()

    def load_password_hash(self):
        if self.SECRET_FILE.exists():
            with open(self.SECRET_FILE) as f:
                data = json.load(f)
            return data.get("password_hash")
        else:
            # Generate a default admin password
            default_hash = hashlib.sha256(b"onyxgodmode").hexdigest()
            with open(self.SECRET_FILE, "w") as f:
                json.dump({"password_hash": default_hash}, f)
            return default_hash

    def authenticate(self, password: str) -> bool:
        if hashlib.sha256(password.encode()).hexdigest() == self.password_hash:
            self.authenticated = True
        return self.authenticated

auth_manager = AuthManager()


# ==============================
# 5. Sandbox Guard
# ==============================
class SandboxGuard:
    """Executes code in isolated Docker containers."""
    def __init__(self):
        import docker
        self.client = docker.from_env()

    def run_python(self, code: str) -> str:
        container = None
        try:
            container = self.client.containers.run(
                image="python:3.11-slim",
                command=["python", "-c", code],
                detach=False,
                remove=True,
                network_mode="none",
                mem_limit="256m",
                cpu_quota=50000,
            )
            return "Executed successfully"
        except Exception as e:
            return f"Sandbox error: {e}"
        finally:
            if container:
                try: container.remove(force=True)
                except: pass

sandbox = SandboxGuard()


# ==============================
# 6. CLI & TUI
# ==============================
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

class OnyxCommandCenter:
    def __init__(self, facade):
        self.facade = facade
        self.session = PromptSession()

    def start(self):
        log_manager.info("[CLI] ONYX Command Center started")
        with patch_stdout():
            while True:
                try:
                    cmd = self.session.prompt("ONYX ▷ ")
                    self.handle_command(cmd)
                except (EOFError, KeyboardInterrupt):
                    break

    def handle_command(self, cmd: str):
        if cmd in ["help", "?"]:
            print("Commands: help, status, backup, exit, sandbox")
        elif cmd == "status":
            print(self.facade.show_system_status())
        elif cmd == "backup":
            self.facade.trigger_backup()
        elif cmd.startswith("sandbox"):
            code = cmd[len("sandbox "):]
            print(sandbox.run_python(code))
        elif cmd == "exit":
            log_manager.info("[CLI] Exiting")
            supervisor.stop_all()
            sys.exit(0)
        else:
            print(f"Unknown command: {cmd}")

# ==============================
# 7. FastAPI Web API
# ==============================
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

@app.get("/api/v1/status")
async def api_status():
    return JSONResponse({"status": "OK", "uptime": time.time() - START_TIME})

@app.get("/api/v1/backups")
async def api_backups(limit: int = 10):
    backups = db.get_backups(limit)
    return JSONResponse({"backups": backups})

def start_api():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ==============================
# 8. System Facade
# ==============================
class SystemFacade:
    """Facade providing unified access to ONYX subsystems."""
    def show_system_status(self):
        return {
            "services": [s.name for s in [docker_manager, vm_manager, plex_manager]],
            "agents": list(cluster_manager.nodes.keys()),
            "backups": len(db.get_backups())
        }

    def trigger_backup(self):
        sources = [Path("/etc"), Path("/home")]
        backup_manager.create_backup(sources)

    def show_ai_status(self):
        return ai_core.get_status() if "ai_core" in globals() else {"epoch":0}

    def run_sandbox_test(self):
        return sandbox.run_python("print('sandbox test')")

facade = SystemFacade()
cli = OnyxCommandCenter(facade)

# ==============================
# PHASE 5: Observability, Games AI, C++ Core, Bash Integration
# ==============================

# ==============================
# 1. Logging & Metrics
# ==============================
import logging
from logging.handlers import TimedRotatingFileHandler
import json

class LogManager:
    def __init__(self, log_file: Path = ONYX_HOME / "onyx.log"):
        self.logger = logging.getLogger("ONYX")
        self.logger.setLevel(logging.INFO)
        handler = TimedRotatingFileHandler(log_file, when="midnight", backupCount=30)
        handler.setFormatter(logging.Formatter('{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}'))
        self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)
        print(f"[INFO] {msg}")

    def error(self, msg):
        self.logger.error(msg)
        print(f"[ERROR] {msg}")

    def get_recent(self, n: int = 20):
        try:
            with open(ONYX_HOME / "onyx.log") as f:
                lines = f.readlines()
            return lines[-n:]
        except Exception as e:
            return [f"[ERROR] Failed to read logs: {e}"]

log_manager = LogManager()


# ==============================
# 2. Supervisor for Threads
# ==============================
class Supervisor:
    def __init__(self):
        self.threads: List[Thread] = []
        self.active = True

    def register_thread(self, t: Thread):
        self.threads.append(t)

    def stop_all(self):
        self.active = False
        for t in self.threads:
            if t.is_alive():
                t.join(timeout=5)

supervisor = Supervisor()


# ==============================
# 3. Games AI
# ==============================
class ChessAI:
    def __init__(self):
        import chess
        import chess.engine
        self.board = chess.Board()
        self.engine_path = "/usr/games/stockfish"
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)

    def make_move(self):
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        self.board.push(result.move)
        return result.move

    def reset(self):
        self.board.reset()

class BlackjackAI:
    def __init__(self):
        self.deck = list(range(1,12))*4
        random.shuffle(self.deck)
        self.hand = []

    def draw(self):
        card = self.deck.pop()
        self.hand.append(card)
        return card

    def score(self):
        return sum(self.hand)

# Placeholder: PokerAI, MinesweeperAI, TicTacToeAI would be similarly implemented

# ==============================
# 4. C++ Core Integration
# ==============================
import ctypes

class CppCore:
    """Load precompiled ONYX C++ shared library."""
    def __init__(self, lib_path: Path = ONYX_HOME / "cpp" / "libonyx_native.so"):
        self.lib = ctypes.CDLL(str(lib_path))
        # Example: bind function `double compute_heavy(double x)`
        self.lib.compute_heavy.argtypes = [ctypes.c_double]
        self.lib.compute_heavy.restype = ctypes.c_double

    def compute_heavy(self, x: float) -> float:
        return self.lib.compute_heavy(x)

cpp_core = CppCore()


# ==============================
# 5. Embedded Bash Scripts
# ==============================
class ShellScripts:
    def __init__(self):
        pass

    def run(self, script: str):
        import subprocess, shlex
        try:
            cmd = shlex.split(script)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"[ERROR] {e}"

bash = ShellScripts()


# ==============================
# 6. Resource & Observability Threads
# ==============================
class MetricsMonitor(Thread):
    def __init__(self):
        super().__init__()
        self.daemon = False
        self.active = True
        self.data = {"cpu": 0, "ram": 0, "disk": 0, "network": 0}

    def run(self):
        import psutil
        while self.active and supervisor.active:
            self.data["cpu"] = psutil.cpu_percent(interval=1)
            self.data["ram"] = psutil.virtual_memory().percent
            self.data["disk"] = psutil.disk_usage("/").percent
            self.data["network"] = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv
            time.sleep(5)

metrics_monitor = MetricsMonitor()
metrics_monitor.start()
supervisor.register_thread(metrics_monitor)


# ==============================
# 7. Integrate Everything into ONYX Core
# ==============================
class ONYXCore:
    def __init__(self):
        self.db = db
        self.backup_manager = backup_manager
        self.auth = auth_manager
        self.sandbox = sandbox
        self.cli = cli
        self.facade = facade
        self.cpp_core = cpp_core
        self.bash = bash
        self.metrics = metrics_monitor
        self.games = {
            "chess": ChessAI(),
            "blackjack": BlackjackAI(),
            # "poker": PokerAI(),
            # "minesweeper": MinesweeperAI(),
            # "tic_tac_toe": TicTacToeAI()
        }

    def start(self):
        log_manager.info("[ONYX] Starting system...")
        self.cli.start()

onyx_core = ONYXCore()


# ==============================
# 8. Clean Shutdown
# ==============================
import atexit, signal

def shutdown(signum=None, frame=None):
    log_manager.info("[ONYX] Shutting down...")
    supervisor.stop_all()
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)
atexit.register(supervisor.stop_all)

# ==============================
# 9. Main Bootstrap
# ==============================
if __name__ == "__main__":
    START_TIME = time.time()
    onyx_core.start()

# ==============================
# PHASE 6: Distributed Agents, Cluster Sync, Reinforcement, Neural Core
# ==============================

# ==============================
# 1. Cluster Node Definition
# ==============================
import socket
import json
import uuid
import psutil

class ClusterNode:
    def __init__(self, host: str = "127.0.0.1", port: int = 5050):
        self.node_id = self._load_or_generate_id()
        self.host = host
        self.port = port
        self.metrics = {"cpu": 0, "ram": 0, "network": 0}
        self.active = True
        self.peers: Dict[str, Dict] = {}

    def _load_or_generate_id(self):
        id_file = ONYX_HOME / "node_id.txt"
        if id_file.exists():
            return id_file.read_text().strip()
        node_id = str(uuid.uuid4())
        id_file.write_text(node_id)
        return node_id

    def update_metrics(self):
        self.metrics["cpu"] = psutil.cpu_percent()
        self.metrics["ram"] = psutil.virtual_memory().percent
        self.metrics["network"] = psutil.net_io_counters().bytes_sent + psutil.net_io_counters().bytes_recv

    def heartbeat(self):
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "metrics": self.metrics,
            "timestamp": time.time()
        }

cluster_node = ClusterNode()


# ==============================
# 2. Distributed Agent
# ==============================
class DistributedAgent(Thread):
    def __init__(self, cluster_node: ClusterNode):
        super().__init__()
        self.cluster_node = cluster_node
        self.active = True
        self.daemon = False

    def run(self):
        while self.active and supervisor.active:
            self.cluster_node.update_metrics()
            self._broadcast_heartbeat()
            time.sleep(5)

    def _broadcast_heartbeat(self):
        for peer_id, peer in self.cluster_node.peers.items():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                s.connect((peer["host"], peer["port"]))
                msg = json.dumps(self.cluster_node.heartbeat()).encode()
                s.sendall(msg)
                s.close()
            except Exception:
                log_manager.error(f"Failed to send heartbeat to {peer_id}")

distributed_agent = DistributedAgent(cluster_node)
distributed_agent.start()
supervisor.register_thread(distributed_agent)


# ==============================
# 3. Predictive Scaler
# ==============================
class PredictiveScaler(Thread):
    def __init__(self, cluster_node: ClusterNode):
        super().__init__()
        self.cluster_node = cluster_node
        self.active = True
        self.daemon = False

    def run(self):
        while self.active and supervisor.active:
            total_cpu = sum(peer.get("metrics", {}).get("cpu", 0) for peer in self.cluster_node.peers.values())
            avg_cpu = (total_cpu + self.cluster_node.metrics["cpu"]) / (len(self.cluster_node.peers)+1)
            if avg_cpu > 80:
                log_manager.info("[Scaler] High CPU detected, activating additional containers/VMs")
                # integrate with Docker/VM Manager
            elif avg_cpu < 20:
                log_manager.info("[Scaler] Low CPU, shutting down idle containers/VMs")
            time.sleep(10)

predictive_scaler = PredictiveScaler(cluster_node)
predictive_scaler.start()
supervisor.register_thread(predictive_scaler)


# ==============================
# 4. Consensus Engine (Raft using pysyncobj)
# ==============================
try:
    from pysyncobj import SyncObj, SyncObjConf, replicated
except ImportError:
    log_manager.error("pysyncobj not installed. Cluster consensus disabled.")
    SyncObj = object
    SyncObjConf = lambda *a, **k: None
    def replicated(func): return func

class ClusterConsensus(SyncObj):
    def __init__(self, self_address, partner_addresses=[]):
        conf = SyncObjConf(dynamicMembershipChange=True)
        super().__init__(self_address, partner_addresses, conf)

    @replicated
    def propose_task(self, task: dict):
        log_manager.info(f"[Consensus] Task proposed: {task}")

consensus_engine = ClusterConsensus(f"{cluster_node.host}:{cluster_node.port}")


# ==============================
# 5. Reinforcement Learning Loop
# ==============================
class RewardSystem:
    def __init__(self):
        self._state = {"total":0, "tasks_completed":0, "idle_learning":0}

    def add_points(self, key:str, points:int):
        self._state[key] = self._state.get(key,0)+points
        self._state["total"] += points

    @property
    def total(self):
        return self._state["total"]

reward_system = RewardSystem()

class MultiAgentRL(Thread):
    def __init__(self):
        super().__init__()
        self.active = True
        self.daemon = False

    def run(self):
        while self.active and supervisor.active:
            # Simplified RL loop
            for agent_name, agent in onyx_core.games.items():
                move = agent.make_move() if hasattr(agent,"make_move") else None
                reward_system.add_points("tasks_completed", 1)
            time.sleep(2)

multi_agent_rl = MultiAgentRL()
multi_agent_rl.start()
supervisor.register_thread(multi_agent_rl)


# ==============================
# 6. Neural Core (PyTorch example)
# ==============================
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    log_manager.error("PyTorch not installed. Neural Core disabled.")

class NeuralCore(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

neural_core = NeuralCore()
optimizer = optim.Adam(neural_core.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


# ==============================
# 7. Integrate Cluster + RL into ONYX Core
# ==============================
onyx_core.cluster_node = cluster_node
onyx_core.distributed_agent = distributed_agent
onyx_core.predictive_scaler = predictive_scaler
onyx_core.consensus_engine = consensus_engine
onyx_core.reward_system = reward_system
onyx_core.multi_agent_rl = multi_agent_rl
onyx_core.neural_core = neural_core
onyx_core.optimizer = optimizer
onyx_core.loss_fn = loss_fn

log_manager.info("[ONYX] Phase 6: Distributed agents, cluster sync, RL, and neural core initialized")

# ==============================
# PHASE 7: API, Web Dashboard, Service Orchestration
# ==============================

# ==============================
# 1. HTTP API & Authentication
# ==============================
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
import pathlib

app = FastAPI(title="ONYX Autonomous System API", version="7.0")

SECRET_KEY = config.get("api_secret", "onyx_secret_key")
JWT_ALGORITHM = "HS256"

security = HTTPBearer()

def create_token(user: str, expires_in: int = 3600):
    payload = {
        "sub": user,
        "exp": datetime.utcnow() + timedelta(seconds=expires_in)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/api/v1/token")
def get_token(username: str, password: str):
    if auth_manager.authenticate(username, password):
        return {"token": create_token(username)}
    raise HTTPException(status_code=403, detail="Invalid credentials")

@app.get("/api/v1/status")
def system_status(user: str = Depends(verify_token)):
    return {
        "system": "ONYX",
        "uptime": time.time() - START_TIME,
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "reward_points": reward_system.total
    }


# ==============================
# 2. Web Dashboard (HTML + Metrics)
# ==============================
from fastapi.responses import HTMLResponse
from jinja2 import Template

dashboard_template = Template("""
<!DOCTYPE html>
<html>
<head>
<title>ONYX Dashboard</title>
<style>
body { font-family: Arial, sans-serif; background: #121212; color: #f0f0f0; }
h1 { color: #00ff99; }
.card { background: #1e1e1e; padding: 10px; margin: 10px; border-radius: 8px; }
</style>
</head>
<body>
<h1>ONYX Dashboard</h1>
<div class="card">
<strong>CPU:</strong> {{ cpu }} %
</div>
<div class="card">
<strong>RAM:</strong> {{ ram }} %
</div>
<div class="card">
<strong>Reward Points:</strong> {{ reward_points }}
</div>
<div class="card">
<strong>Services:</strong>
<ul>
{% for svc, status in services.items() %}
<li>{{ svc }}: {{ status }}</li>
{% endfor %}
</ul>
</div>
</body>
</html>
""")

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    services_status = {svc.name: "Running" if svc.is_active() else "Stopped" for svc in service_manager.services}
    return dashboard_template.render(
        cpu=psutil.cpu_percent(),
        ram=psutil.virtual_memory().percent,
        reward_points=reward_system.total,
        services=services_status
    )


# ==============================
# 3. WebSocket for live metrics
# ==============================
from fastapi import WebSocket
from fastapi.responses import JSONResponse

@app.websocket("/ws/metrics")
async def websocket_metrics(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = {
                "cpu": psutil.cpu_percent(),
                "ram": psutil.virtual_memory().percent,
                "reward_points": reward_system.total
            }
            await ws.send_json(data)
            await asyncio.sleep(2)
    except Exception:
        await ws.close()


# ==============================
# 4. Service Manager
# ==============================
class ServiceManager:
    def __init__(self):
        self.services = []

    def register_service(self, service):
        self.services.append(service)

    def start_all(self):
        for svc in self.services:
            svc.start()

    def stop_all(self):
        for svc in self.services:
            svc.stop()

    def get_status(self):
        return {svc.name: svc.is_active() for svc in self.services}

service_manager = ServiceManager()


# ==============================
# 5. Service Example (Docker)
# ==============================
import docker

class DockerService:
    def __init__(self, name: str, image: str):
        self.name = name
        self.image = image
        self.client = docker.from_env()
        self.container = None

    def start(self):
        if not self.container:
            self.container = self.client.containers.run(self.image, name=self.name, detach=True)
        elif self.container.status != "running":
            self.container.start()

    def stop(self):
        if self.container:
            self.container.stop()

    def is_active(self):
        if not self.container:
            return False
        self.container.reload()
        return self.container.status == "running"

docker_service = DockerService("test_container", "python:3.11-slim")
service_manager.register_service(docker_service)
service_manager.start_all()


# ==============================
# 6. Backup Orchestration
# ==============================
import tarfile

class BackupOrchestrator(Thread):
    def __init__(self, directories: list, backup_dir: str):
        super().__init__()
        self.directories = directories
        self.backup_dir = pathlib.Path(backup_dir)
        self.active = True
        self.daemon = False

    def run(self):
        while self.active and supervisor.active:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"backup_{timestamp}.tar.gz"
            with tarfile.open(backup_file, "w:gz") as tar:
                for d in self.directories:
                    tar.add(d, arcname=pathlib.Path(d).name)
            log_manager.info(f"[Backup] Backup created: {backup_file}")
            time.sleep(3600)  # hourly backup

backup_orchestrator = BackupOrchestrator(directories=["/etc", "/home"], backup_dir="~/.onyx/backups")
backup_orchestrator.start()
supervisor.register_thread(backup_orchestrator)


# ==============================
# 7. IDS / IPS Integration (simplified)
# ==============================
from scapy.all import sniff, IP

class IntrusionDetection(Thread):
    def __init__(self, threshold=100):
        super().__init__()
        self.threshold = threshold
        self.active = True
        self.daemon = False
        self.alerts = []

    def run(self):
        sniff(prn=self.process_packet, store=False, stop_filter=lambda _: not self.active)

    def process_packet(self, pkt):
        if IP in pkt:
            src = pkt[IP].src
            self.alerts.append((src, time.time()))
            if len(self.alerts) > self.threshold:
                log_manager.warn(f"[IDS] High traffic from {src}")
                self.alerts = self.alerts[-self.threshold:]

ids_engine = IntrusionDetection()
ids_engine.start()
supervisor.register_thread(ids_engine)

log_manager.info("[ONYX] Phase 7: API, Dashboard, Services, Backup, IDS initialized")

# ==============================
# PHASE 8: Games AI + Analytics + Neural Optimization
# ==============================

import random
import numpy as np
import chess
import chess.engine
import json

# ==============================
# 1. Reward System
# ==============================
class RewardSystem:
    def __init__(self):
        self._points = 0
        self.lock = threading.Lock()

    def add(self, value: int):
        with self.lock:
            self._points += value

    def get_total(self):
        with self.lock:
            return self._points

reward_system = RewardSystem()


# ==============================
# 2. Chess AI Agent
# ==============================
class ChessAgent:
    def __init__(self, skill_level=1):
        self.board = chess.Board()
        self.engine = None
        self.skill_level = skill_level  # 1=easy, 5=hard
        self.lock = threading.Lock()

    def set_engine(self, path: str):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(path)
        except Exception as e:
            log_manager.error(f"[ChessAgent] Engine failed: {e}")

    def make_move(self):
        with self.lock:
            if self.board.is_game_over():
                return None
            try:
                result = self.engine.play(self.board, chess.engine.Limit(time=0.1 * self.skill_level))
                self.board.push(result.move)
                reward_system.add(5)  # reward points per move
                return result.move
            except Exception as e:
                log_manager.error(f"[ChessAgent] Move failed: {e}")
                return None

    def restart_game(self):
        with self.lock:
            self.board.reset()


chess_agent = ChessAgent(skill_level=2)
chess_agent.set_engine("/usr/bin/stockfish")


# ==============================
# 3. Blackjack AI
# ==============================
class BlackjackAgent:
    def __init__(self):
        self.hand = []
        self.deck = self.init_deck()
        self.lock = threading.Lock()

    def init_deck(self):
        suits = ["H", "D", "C", "S"]
        ranks = list(range(1, 14))
        deck = [(r, s) for r in ranks for s in suits] * 6
        random.shuffle(deck)
        return deck

    def draw_card(self):
        if not self.deck:
            self.deck = self.init_deck()
        return self.deck.pop()

    def play_hand(self):
        self.hand = [self.draw_card(), self.draw_card()]
        total = self.calc_total()
        while total < 17:
            self.hand.append(self.draw_card())
            total = self.calc_total()
        if total <= 21:
            reward_system.add(10)
        return total

    def calc_total(self):
        total = 0
        aces = 0
        for r, s in self.hand:
            if r > 10:
                total += 10
            elif r == 1:
                total += 11
                aces += 1
            else:
                total += r
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

blackjack_agent = BlackjackAgent()


# ==============================
# 4. Tic-Tac-Toe AI
# ==============================
class TicTacToeAgent:
    def __init__(self):
        self.board = [" "] * 9
        self.lock = threading.Lock()

    def make_move(self, symbol="X"):
        with self.lock:
            empty = [i for i, v in enumerate(self.board) if v == " "]
            if not empty:
                return None
            move = random.choice(empty)
            self.board[move] = symbol
            reward_system.add(2)
            return move

    def restart(self):
        with self.lock:
            self.board = [" "] * 9

tic_tac_toe_agent = TicTacToeAgent()


# ==============================
# 5. Reinforcement Learning Framework
# ==============================
class RLAgent(Thread):
    def __init__(self, game_agent, episodes=1000):
        super().__init__()
        self.game_agent = game_agent
        self.episodes = episodes
        self.active = True
        self.daemon = False

    def run(self):
        for _ in range(self.episodes):
            if not self.active:
                break
            if isinstance(self.game_agent, BlackjackAgent):
                total = self.game_agent.play_hand()
                reward_system.add(total // 2)
            elif isinstance(self.game_agent, TicTacToeAgent):
                self.game_agent.make_move()
            elif isinstance(self.game_agent, ChessAgent):
                self.game_agent.make_move()
            time.sleep(0.05)  # simulate thinking time

    def stop(self):
        self.active = False

# Start RL agents
rl_chess = RLAgent(chess_agent, episodes=500)
rl_blackjack = RLAgent(blackjack_agent, episodes=1000)
rl_tictactoe = RLAgent(tic_tac_toe_agent, episodes=800)

supervisor.register_thread(rl_chess)
supervisor.register_thread(rl_blackjack)
supervisor.register_thread(rl_tictactoe)

rl_chess.start()
rl_blackjack.start()
rl_tictactoe.start()


# ==============================
# 6. Analytics Engine
# ==============================
class AnalyticsEngine:
    def __init__(self):
        self.data = []
        self.lock = threading.Lock()

    def record_metrics(self):
        with self.lock:
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": psutil.cpu_percent(),
                "ram": psutil.virtual_memory().percent,
                "reward_points": reward_system.get_total()
            }
            self.data.append(entry)

    def export_json(self, path="~/.onyx/analytics.json"):
        with self.lock:
            pathlib.Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            with open(pathlib.Path(path).expanduser(), "w") as f:
                json.dump(self.data, f, indent=2)

analytics_engine = AnalyticsEngine()


# ==============================
# 7. Neural Core for Predictions
# ==============================
class NeuralCore(Thread):
    def __init__(self):
        super().__init__()
        self.active = True
        self.daemon = False
        self.weights = np.random.rand(10, 10)
        self.lock = threading.Lock()

    def run(self):
        while self.active:
            self.optimize()
            time.sleep(1)

    def optimize(self):
        with self.lock:
            gradient = np.random.randn(10, 10) * 0.01
            self.weights -= gradient
            reward_system.add(1)

    def predict(self, inputs: np.ndarray):
        with self.lock:
            return np.dot(inputs, self.weights)

neural_core = NeuralCore()
supervisor.register_thread(neural_core)
neural_core.start()

log_manager.info("[ONYX] Phase 8: Games AI, RL, Analytics, Neural Core initialized")

# ==============================
# PHASE 9: Cluster + Distributed Agents + Consensus
# ==============================

import socket
import struct
import json
import select
import random
import uuid

# ==============================
# 1. Cluster Node Representation
# ==============================
class ClusterNode:
    def __init__(self, host="127.0.0.1", port=5050):
        self.node_id = self._generate_node_id()
        self.host = host
        self.port = port
        self.metrics = {}
        self.last_heartbeat = datetime.utcnow()
        self.lock = threading.Lock()

    def _generate_node_id(self):
        # persistent UUID from host MAC
        mac = uuid.getnode()
        return str(uuid.UUID(int=mac << 80 | random.getrandbits(80)))

    def update_metrics(self):
        with self.lock:
            self.metrics = {
                "cpu": psutil.cpu_percent(),
                "ram": psutil.virtual_memory().percent,
                "uptime": time.time() - psutil.boot_time()
            }
            self.last_heartbeat = datetime.utcnow()

cluster_node = ClusterNode()


# ==============================
# 2. Cluster Manager
# ==============================
class ClusterManager(Thread):
    def __init__(self, node: ClusterNode, nodes_list=None):
        super().__init__()
        self.node = node
        self.nodes_list = nodes_list if nodes_list else [self.node]
        self.active = True
        self.lock = threading.Lock()
        self.daemon = False

    def run(self):
        while self.active:
            self.node.update_metrics()
            self.broadcast_heartbeat()
            self.collect_heartbeats()
            time.sleep(5)

    def broadcast_heartbeat(self):
        payload = json.dumps({
            "node_id": self.node.node_id,
            "metrics": self.node.metrics,
            "timestamp": datetime.utcnow().isoformat()
        }).encode()
        with self.lock:
            for peer in self.nodes_list:
                if peer.node_id == self.node.node_id:
                    continue  # skip self
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.settimeout(1)
                    sock.sendto(payload, (peer.host, peer.port))
                    sock.close()
                except Exception as e:
                    log_manager.error(f"[ClusterManager] Heartbeat failed: {e}")

    def collect_heartbeats(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.node.host, self.node.port))
        sock.settimeout(1)
        try:
            while True:
                try:
                    data, addr = sock.recvfrom(4096)
                    message = json.loads(data.decode())
                    self._update_peer(message)
                except socket.timeout:
                    break
        finally:
            sock.close()

    def _update_peer(self, message):
        node_id = message.get("node_id")
        metrics = message.get("metrics")
        ts = message.get("timestamp")
        with self.lock:
            for peer in self.nodes_list:
                if peer.node_id == node_id:
                    peer.metrics = metrics
                    peer.last_heartbeat = datetime.fromisoformat(ts)
                    return
            # if unknown node, add
            new_peer = ClusterNode()
            new_peer.node_id = node_id
            new_peer.host = "unknown"
            new_peer.port = 5050
            new_peer.metrics = metrics
            new_peer.last_heartbeat = datetime.fromisoformat(ts)
            self.nodes_list.append(new_peer)

    def stop(self):
        self.active = False

cluster_manager = ClusterManager(cluster_node)
supervisor.register_thread(cluster_manager)
cluster_manager.start()


# ==============================
# 3. Distributed Agent
# ==============================
class DistributedAgent(Thread):
    def __init__(self, agent_id, task_queue, cluster_manager: ClusterManager):
        super().__init__()
        self.agent_id = agent_id
        self.task_queue = task_queue
        self.cluster_manager = cluster_manager
        self.active = True
        self.daemon = False

    def run(self):
        while self.active:
            if not self.task_queue:
                time.sleep(0.1)
                continue
            task = self.task_queue.pop(0)
            result = self.execute_task(task)
            log_manager.info(f"[DistributedAgent-{self.agent_id}] Task {task['name']} result: {result}")
            reward_system.add(1)
            time.sleep(0.05)

    def execute_task(self, task):
        # simple simulated execution
        if task["type"] == "backup":
            return f"Backup completed at {datetime.utcnow().isoformat()}"
        elif task["type"] == "compute":
            return sum(range(1000))
        return "Unknown task"

    def stop(self):
        self.active = False

task_queue = [{"name": f"task-{i}", "type": random.choice(["backup","compute"])} for i in range(20)]
dist_agent = DistributedAgent(agent_id="agent-001", task_queue=task_queue, cluster_manager=cluster_manager)
supervisor.register_thread(dist_agent)
dist_agent.start()


# ==============================
# 4. Consensus Engine (Simplified)
# ==============================
class ConsensusEngine:
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager = cluster_manager

    def propose(self, value):
        votes = 0
        for peer in self.cluster_manager.nodes_list:
            if random.random() > 0.2:  # 80% simulated agreement
                votes += 1
        majority = len(self.cluster_manager.nodes_list) // 2 + 1
        approved = votes >= majority
        log_manager.info(f"[ConsensusEngine] Proposed value '{value}' approved={approved} ({votes}/{len(self.cluster_manager.nodes_list)})")
        return approved

consensus_engine = ConsensusEngine(cluster_manager)

# Example of proposing a config change
consensus_engine.propose("increase_backup_frequency")

log_manager.info("[ONYX] Phase 9: Cluster and Distributed Agents initialized")

# ==============================
# PHASE 10: API, Web Dashboard, CLI Enhancements, Observability
# ==============================

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from prometheus_client import start_http_server, Counter, Gauge, generate_latest
import asyncio

# ==============================
# 1. Prometheus Metrics
# ==============================
PROM_CPU = Gauge("onyx_cpu_percent", "CPU Usage %")
PROM_RAM = Gauge("onyx_ram_percent", "RAM Usage %")
PROM_TASKS = Gauge("onyx_task_queue_length", "Length of task queue")
PROM_REWARDS = Gauge("onyx_reward_points", "Total reward points")

def update_prom_metrics():
    while True:
        PROM_CPU.set(psutil.cpu_percent())
        PROM_RAM.set(psutil.virtual_memory().percent)
        PROM_TASKS.set(len(task_queue))
        PROM_REWARDS.set(reward_system.total)
        time.sleep(5)

prom_thread = Thread(target=update_prom_metrics)
supervisor.register_thread(prom_thread)
prom_thread.start()
start_http_server(8001)  # Expose Prometheus metrics

# ==============================
# 2. FastAPI App
# ==============================
app = FastAPI(title="ONYX Monitoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/status")
def api_status():
    return {
        "system": "ONYX",
        "uptime": time.time() - psutil.boot_time(),
        "agents": len(supervisor.threads),
        "tasks_pending": len(task_queue),
        "reward_points": reward_system.total,
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent
    }

@app.get("/api/v1/metrics")
def api_metrics():
    return JSONResponse(content={
        "cpu": PROM_CPU._value.get(),
        "ram": PROM_RAM._value.get(),
        "tasks": PROM_TASKS._value.get(),
        "rewards": PROM_REWARDS._value.get()
    })

@app.post("/api/v1/task")
def api_add_task(task: dict):
    task_queue.append(task)
    return {"status": "queued", "task": task}

@app.get("/api/v1/consensus/{value}")
def api_consensus(value: str):
    approved = consensus_engine.propose(value)
    return {"value": value, "approved": approved}

# ==============================
# 3. WebSocket Event Server
# ==============================
class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

ws_manager = WebSocketManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await ws_manager.broadcast({"event": "echo", "data": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# ==============================
# 4. Enhanced CLI / TUI
# ==============================
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

class OnyxCLI:
    def __init__(self, facade):
        self.facade = facade
        self.session = PromptSession()
        self.commands = {
            "status": self.show_status,
            "tasks": self.show_tasks,
            "agents": self.show_agents,
            "backup": self.trigger_backup,
            "exit": self.exit_cli
        }

    def show_status(self, args=None):
        s = self.facade.show_system_status()
        console.print_json(json.dumps(s))

    def show_tasks(self, args=None):
        console.print(f"Pending tasks: {len(task_queue)}")
        for t in task_queue:
            console.print(f"- {t}")

    def show_agents(self, args=None):
        console.print(f"Active agents: {len(supervisor.threads)}")
        for t in supervisor.threads:
            console.print(f"- {t.name}")

    def trigger_backup(self, args=None):
        self.facade.trigger_backup()
        console.print("[ONYX] Backup triggered.")

    def exit_cli(self, args=None):
        console.print("[ONYX] Exiting CLI...")
        supervisor.stop_all()
        sys.exit(0)

    def start(self):
        with patch_stdout():
            while True:
                try:
                    cmd_line = self.session.prompt("ONYX ▷ ")
                    if not cmd_line.strip():
                        continue
                    parts = cmd_line.strip().split()
                    cmd = parts[0]
                    args = parts[1:] if len(parts) > 1 else None
                    func = self.commands.get(cmd, None)
                    if func:
                        func(args)
                    else:
                        console.print(f"[ERROR] Unknown command: {cmd}")
                except KeyboardInterrupt:
                    console.print("[ONYX] CTRL-C detected. Exiting...")
                    self.exit_cli()

# ==============================
# 5. System Facade Integration
# ==============================
class SystemFacade:
    def show_system_status(self):
        return {
            "uptime": time.time() - psutil.boot_time(),
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent,
            "tasks_pending": len(task_queue),
            "reward_points": reward_system.total
        }

    def trigger_backup(self):
        backup_manager.perform_backup()

facade = SystemFacade()

# ==============================
# 6. Main Entrypoint
# ==============================
if __name__ == "__main__":
    # Start API in background thread
    api_thread = Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info"))
    supervisor.register_thread(api_thread)
    api_thread.start()

    # Launch CLI
    cli = OnyxCLI(facade)
    cli.start()

# ==============================
# PHASE 11: Game AI + Reinforcement + Hybrid Sandbox + C++ Integration
# ==============================

import ctypes
import tempfile
import subprocess
import random
import chess  # python-chess
import chess.engine
import numpy as np

# ==============================
# 1. C++ Core Bindings (ctypes)
# ==============================
# For demonstration, we embed a small C++ optimization routine
cpp_code = r"""
#include <cmath>
#include <vector>
extern "C" {
    double sum_squares(double* arr, int n) {
        double sum = 0.0;
        for(int i=0;i<n;i++) sum += arr[i]*arr[i];
        return sum;
    }
}
"""
cpp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".cpp")
cpp_file.write(cpp_code.encode())
cpp_file.close()
lib_file = cpp_file.name.replace(".cpp",".so")
subprocess.run(["g++","-shared","-fPIC","-O3",cpp_file.name,"-o",lib_file], check=True)
cpp_lib = ctypes.CDLL(lib_file)
cpp_lib.sum_squares.restype = ctypes.c_double
cpp_lib.sum_squares.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]

def sum_squares_py(arr: list[float]) -> float:
    arr_ctypes = (ctypes.c_double * len(arr))(*arr)
    return cpp_lib.sum_squares(arr_ctypes, len(arr))

# ==============================
# 2. Hybrid Sandbox Execution
# ==============================
class HybridSandbox:
    def run_python(self, code: str):
        """Run arbitrary Python code safely inside a subprocess"""
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip(), result.stderr.strip()

    def run_cpp(self, code: str):
        """Run C++ code in temporary compilation sandbox"""
        cpp_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cpp")
        cpp_tmp.write(code.encode())
        cpp_tmp.close()
        bin_tmp = cpp_tmp.name.replace(".cpp",".out")
        subprocess.run(["g++","-O3",cpp_tmp.name,"-o",bin_tmp], check=True)
        result = subprocess.run([bin_tmp], capture_output=True, text=True)
        os.remove(cpp_tmp.name)
        os.remove(bin_tmp)
        return result.stdout.strip()

sandbox = HybridSandbox()

# ==============================
# 3. Game AI Agents
# ==============================
class ChessAI:
    def __init__(self, engine_path: str = "/usr/bin/stockfish"):
        self.engine_path = engine_path
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)

    def best_move(self, board: chess.Board, time_limit: float = 0.1):
        result = self.engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move

    def shutdown(self):
        self.engine.quit()

# Blackjack AI
class BlackjackAI:
    def decide(self, hand_total: int, dealer_card: int):
        # Very simple policy: hit < 17
        return "hit" if hand_total < 17 else "stand"

# Tic-Tac-Toe AI
class TicTacToeAI:
    def __init__(self):
        self.board = [" "]*9

    def available_moves(self):
        return [i for i,x in enumerate(self.board) if x==" "]

    def move(self, player="X"):
        moves = self.available_moves()
        return random.choice(moves) if moves else None

# ==============================
# 4. Reinforcement Loop
# ==============================
class RewardSystem:
    def __init__(self):
        self._state = {"total":0, "games_played":0}

    @property
    def total(self):
        return self._state["total"]

    def add_points(self, points: int):
        self._state["total"] += points
        self._state["games_played"] += 1

reward_system = RewardSystem()

def play_chess_game():
    board = chess.Board()
    ai = ChessAI()
    while not board.is_game_over():
        move = ai.best_move(board)
        board.push(move)
    reward_system.add_points(10 if board.result()=="1-0" else 5)
    ai.shutdown()

def play_blackjack_game():
    ai = BlackjackAI()
    hand = [random.randint(1,11), random.randint(1,11)]
    dealer = random.randint(1,11)
    while ai.decide(sum(hand), dealer)=="hit":
        hand.append(random.randint(1,11))
    reward_system.add_points(sum(hand))

# ==============================
# 5. Task Queue Integration
# ==============================
task_queue.append({"type":"chess_game", "func":play_chess_game})
task_queue.append({"type":"blackjack", "func":play_blackjack_game})

def game_loop():
    while True:
        if task_queue:
            task = task_queue.pop(0)
            try:
                task["func"]()
            except Exception as e:
                console.print(f"[ERROR] Task failed: {e}")
        time.sleep(1)

game_thread = Thread(target=game_loop)
supervisor.register_thread(game_thread)
game_thread.start()

# ==============================
# 6. Example Usage of C++ Core
# ==============================
arr = [1.0,2.0,3.0,4.0,5.0]
result = sum_squares_py(arr)
console.print(f"[ONYX] C++ sum_squares({arr}) = {result}")

# ==============================
# PHASE 12: Distributed Cluster + Consensus + Multi-Agent Orchestration
# ==============================

import socket
import json
import uuid
import platform

# ==============================
# 1. Cluster Node
# ==============================
class ClusterNode:
    def __init__(self, host="127.0.0.1", port=5050):
        self.node_id = self._load_or_generate_id()
        self.host = host
        self.port = port
        self.metrics = {"cpu":0.0, "ram":0.0, "tasks":0}
        self.active = True

    def _load_or_generate_id(self):
        id_file = os.path.expanduser("~/.onyx/node_id")
        if os.path.exists(id_file):
            with open(id_file,"r") as f:
                return f.read().strip()
        else:
            node_id = str(uuid.uuid4())
            os.makedirs(os.path.dirname(id_file), exist_ok=True)
            with open(id_file,"w") as f:
                f.write(node_id)
            return node_id

    def update_metrics(self):
        self.metrics["cpu"] = psutil.cpu_percent()
        self.metrics["ram"] = psutil.virtual_memory().percent
        self.metrics["tasks"] = len(task_queue)
        self.metrics["uptime"] = time.time() - psutil.boot_time()

# ==============================
# 2. Cluster Manager
# ==============================
class ClusterManager:
    def __init__(self):
        self.nodes = {}  # node_id: {"host":..., "port":..., "metrics":...}
        self.active = True

    def register_node(self, node: ClusterNode):
        self.nodes[node.node_id] = {
            "host": node.host,
            "port": node.port,
            "metrics": node.metrics
        }

    def broadcast_metrics(self, node: ClusterNode):
        data = json.dumps({
            "node_id": node.node_id,
            "metrics": node.metrics
        }).encode()
        for n_id, info in self.nodes.items():
            if n_id == node.node_id: 
                continue
            try:
                with socket.create_connection((info["host"], info["port"]), timeout=1) as s:
                    s.sendall(data)
            except Exception as e:
                console.print(f"[CLUSTER] Failed to send to {info['host']}:{info['port']} - {e}")

    def consensus_decision(self):
        """Simple majority voting across nodes based on CPU load for a demo"""
        scores = [info["metrics"].get("cpu", 100.0) for info in self.nodes.values()]
        avg_cpu = sum(scores)/len(scores) if scores else 0.0
        decision = "scale_up" if avg_cpu > 70 else "scale_down"
        return decision

# ==============================
# 3. Node Heartbeat Loop
# ==============================
def heartbeat_loop(node: ClusterNode, manager: ClusterManager, interval: float = 5.0):
    while node.active:
        node.update_metrics()
        manager.register_node(node)
        manager.broadcast_metrics(node)
        time.sleep(interval)

# ==============================
# 4. Distributed Task Dispatcher
# ==============================
class DistributedAgent:
    def __init__(self, node: ClusterNode, cluster: ClusterManager):
        self.node = node
        self.cluster = cluster

    def dispatch_task(self, task: dict):
        # Find node with lowest CPU load
        min_cpu = float("inf")
        target_node_id = None
        for n_id, info in self.cluster.nodes.items():
            cpu = info["metrics"].get("cpu", 100)
            if cpu < min_cpu:
                min_cpu = cpu
                target_node_id = n_id
        if target_node_id == self.node.node_id:
            task_queue.append(task)
            console.print(f"[TASK] Executing locally: {task['type']}")
        else:
            # Send task over network
            info = self.cluster.nodes[target_node_id]
            try:
                data = json.dumps(task).encode()
                with socket.create_connection((info["host"], info["port"]), timeout=1) as s:
                    s.sendall(data)
                console.print(f"[TASK] Dispatched to {info['host']}")
            except Exception as e:
                console.print(f"[TASK] Failed to dispatch: {e}")

# ==============================
# 5. Supervisor Integration
# ==============================
local_node = ClusterNode()
cluster_manager = ClusterManager()
distributed_agent = DistributedAgent(local_node, cluster_manager)

heartbeat_thread = Thread(target=heartbeat_loop, args=(local_node, cluster_manager))
supervisor.register_thread(heartbeat_thread)
heartbeat_thread.start()

# ==============================
# 6. Consensus Orchestrator
# ==============================
def consensus_loop(cluster: ClusterManager):
    while True:
        decision = cluster.consensus_decision()
        if decision == "scale_up":
            console.print("[CONSENSUS] Scaling up resources...")
            # Could trigger additional VM/Service allocation
        else:
            console.print("[CONSENSUS] Scaling down resources...")
        time.sleep(10)

consensus_thread = Thread(target=consensus_loop, args=(cluster_manager,))
supervisor.register_thread(consensus_thread)
consensus_thread.start()

# ==============================
# 7. Example Task Dispatch
# ==============================
sample_task = {"type":"chess_game","func":play_chess_game}
distributed_agent.dispatch_task(sample_task)

console.print(f"[ONYX] Cluster nodes registered: {list(cluster_manager.nodes.keys())}")

# ==============================
# PHASE 13: Full API + Web Dashboard + WebSocket Integration
# ==============================

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ==============================
# 1. FastAPI app
# ==============================
app = FastAPI(title="ONYX Remote API", version="1.0")

# ==============================
# 2. Simple in-memory state
# ==============================
connected_websockets = set()

# ==============================
# 3. API Endpoints
# ==============================
@app.get("/api/v1/status")
async def get_status():
    return JSONResponse(content={
        "node_id": local_node.node_id,
        "cpu": local_node.metrics.get("cpu"),
        "ram": local_node.metrics.get("ram"),
        "tasks": len(task_queue),
        "uptime": local_node.metrics.get("uptime"),
        "cluster_nodes": list(cluster_manager.nodes.keys())
    })

@app.get("/api/v1/tasks")
async def get_tasks():
    return JSONResponse(content={"queue": [t["type"] for t in task_queue]})

@app.post("/api/v1/tasks")
async def submit_task(task: dict):
    distributed_agent.dispatch_task(task)
    return JSONResponse(content={"status":"submitted","task":task})

@app.get("/api/v1/ai-status")
async def ai_status():
    return JSONResponse(content={
        "epoch": ai_core.epoch,
        "last_loss": ai_core.last_loss
    })

# ==============================
# 4. WebSocket endpoint for live metrics
# ==============================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
            data = {
                "cpu": local_node.metrics.get("cpu"),
                "ram": local_node.metrics.get("ram"),
                "tasks": len(task_queue)
            }
            await websocket.send_json(data)
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)

# ==============================
# 5. Periodic broadcaster for cluster updates
# ==============================
async def broadcast_cluster_updates():
    while True:
        if connected_websockets:
            data = {
                "cluster_nodes": list(cluster_manager.nodes.keys()),
                "metrics": {nid: n["metrics"] for nid, n in cluster_manager.nodes.items()}
            }
            websockets_to_remove = set()
            for ws in connected_websockets:
                try:
                    await ws.send_json(data)
                except:
                    websockets_to_remove.add(ws)
            connected_websockets.difference_update(websockets_to_remove)
        await asyncio.sleep(5)

# ==============================
# 6. Web dashboard (static HTML/JS)
# ==============================
HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
<title>ONYX Dashboard</title>
<style>
body{font-family:sans-serif;background:#1e1e1e;color:#f5f5f5;padding:20px;}
#metrics{display:flex;gap:20px;}
.metric{padding:10px;background:#2e2e2e;border-radius:5px;}
</style>
</head>
<body>
<h1>ONYX Dashboard</h1>
<div id="metrics">
  <div class="metric">CPU: <span id="cpu">0</span>%</div>
  <div class="metric">RAM: <span id="ram">0</span>%</div>
  <div class="metric">Tasks: <span id="tasks">0</span></div>
  <div class="metric">Nodes: <span id="nodes">0</span></div>
</div>
<script>
let ws = new WebSocket("ws://"+location.host+"/ws");
ws.onmessage = (event)=>{
  let data = JSON.parse(event.data);
  if(data.cpu!==undefined) document.getElementById("cpu").textContent = data.cpu;
  if(data.ram!==undefined) document.getElementById("ram").textContent = data.ram;
  if(data.tasks!==undefined) document.getElementById("tasks").textContent = data.tasks;
  if(data.cluster_nodes!==undefined) document.getElementById("nodes").textContent = data.cluster_nodes.length;
}
</script>
</body>
</html>
"""

@app.get("/")
async def dashboard():
    return HTMLResponse(content=HTML_DASHBOARD)

# ==============================
# 7. Run FastAPI server in a thread
# ==============================
def start_api():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(broadcast_cluster_updates())
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

api_thread = Thread(target=start_api)
supervisor.register_thread(api_thread)
api_thread.start()

console.print("[ONYX] API + WebSocket server running at http://localhost:8000")

# ==============================
# PHASE 14: Game AI Integration + Reinforcement Learning
# ==============================

import random
import chess
import pickle
from collections import deque
from enum import Enum

# ==============================
# 1. Game Types
# ==============================
class GameType(Enum):
    CHESS = "chess"
    BLACKJACK = "blackjack"
    POKER = "poker"
    MINESWEEPER = "minesweeper"
    TIC_TAC_TOE = "tic_tac_toe"
    ARENA = "arena"

# ==============================
# 2. Persistent Game Memory
# ==============================
class GameMemory:
    def __init__(self, db_file="game_memory.pkl"):
        self.db_file = db_file
        try:
            with open(self.db_file, "rb") as f:
                self.memory = pickle.load(f)
        except FileNotFoundError:
            self.memory = {}

    def save(self):
        with open(self.db_file, "wb") as f:
            pickle.dump(self.memory, f)

    def store(self, game_type, state, reward):
        if game_type not in self.memory:
            self.memory[game_type] = []
        self.memory[game_type].append({"state": state, "reward": reward})
        self.save()

    def retrieve(self, game_type):
        return self.memory.get(game_type, [])

game_memory = GameMemory()

# ==============================
# 3. Chess AI
# ==============================
class ChessAI:
    def __init__(self):
        self.board = chess.Board()

    def move(self):
        legal_moves = list(self.board.legal_moves)
        move = random.choice(legal_moves)
        self.board.push(move)
        return move

    def play(self, turns=10):
        for _ in range(turns):
            move = self.move()
            # store board state and reward (simplified reward)
            reward = 1 if not self.board.is_checkmate() else 10
            game_memory.store(GameType.CHESS.value, self.board.fen(), reward)

chess_ai = ChessAI()

# ==============================
# 4. Blackjack AI
# ==============================
class BlackjackAI:
    def __init__(self):
        self.deck = [2,3,4,5,6,7,8,9,10,10,10,10,11]*4
        self.hand = []

    def draw_card(self):
        return self.deck.pop(random.randint(0, len(self.deck)-1))

    def play_hand(self):
        self.hand = [self.draw_card(), self.draw_card()]
        total = sum(self.hand)
        while total < 17:
            self.hand.append(self.draw_card())
            total = sum(self.hand)
        reward = 1 if total <= 21 else -1
        game_memory.store(GameType.BLACKJACK.value, self.hand[:], reward)
        return total

blackjack_ai = BlackjackAI()

# ==============================
# 5. Tic-Tac-Toe AI
# ==============================
class TicTacToeAI:
    def __init__(self):
        self.board = [" "]*9

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x==" "]

    def move(self, mark="X"):
        moves = self.available_moves()
        choice = random.choice(moves)
        self.board[choice] = mark
        reward = 0
        if self.check_win(mark):
            reward = 10
        game_memory.store(GameType.TIC_TAC_TOE.value, self.board[:], reward)
        return choice

    def check_win(self, mark):
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        return any(all(self.board[i]==mark for i in line) for line in wins)

tic_tac_toe_ai = TicTacToeAI()

# ==============================
# 6. Arena AI (multi-game RL loop)
# ==============================
class Arena:
    def __init__(self):
        self.games = [chess_ai, blackjack_ai, tic_tac_toe_ai]
        self.queue = deque(maxlen=100)

    def simulate(self):
        for game in self.games:
            if isinstance(game, ChessAI):
                game.play(turns=5)
            elif isinstance(game, BlackjackAI):
                game.play_hand()
            elif isinstance(game, TicTacToeAI):
                game.move()
            # collect rewards
            self.queue.append(game_memory.retrieve(game.__class__.__name__.lower()))

arena = Arena()

# ==============================
# 7. Scheduler Integration
# ==============================
def game_loop():
    while True:
        arena.simulate()
        time.sleep(5)

game_thread = Thread(target=game_loop)
supervisor.register_thread(game_thread)
game_thread.start()

console.print("[ONYX] Game AI simulations running (Chess, Blackjack, Tic-Tac-Toe, Arena)")

# ==============================
# PHASE 15: Cluster Sync + Predictive Scaling + Distributed Agents
# ==============================

import socket
import json
import uuid
import platform
from collections import defaultdict

# ==============================
# 1. Cluster Node
# ==============================
class ClusterNode:
    def __init__(self, host=None, port=5050):
        self.node_id = self.generate_node_id()
        self.host = host or self.get_local_ip()
        self.port = port
        self.metrics = {"cpu": 0, "ram": 0, "tasks": 0, "uptime": 0}
        self.last_heartbeat = time.time()
        self.peers = {}

    def generate_node_id(self):
        # persistent node id stored in ~/.onyx/node_id
        node_file = os.path.expanduser("~/.onyx/node_id")
        try:
            with open(node_file, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            node_id = str(uuid.uuid4())
            os.makedirs(os.path.dirname(node_file), exist_ok=True)
            with open(node_file, "w") as f:
                f.write(node_id)
            return node_id

    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    def update_metrics(self):
        self.metrics["cpu"] = psutil.cpu_percent()
        self.metrics["ram"] = psutil.virtual_memory().percent
        self.metrics["uptime"] = time.time() - psutil.boot_time()
        self.metrics["tasks"] = len(supervisor.threads)
        self.last_heartbeat = time.time()

# ==============================
# 2. Cluster Manager
# ==============================
class ClusterManager:
    def __init__(self, node):
        self.node = node
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(2.0)
        self.running = True
        self.sync_interval = 10  # seconds

    def broadcast_heartbeat(self):
        msg = {
            "node_id": self.node.node_id,
            "host": self.node.host,
            "port": self.node.port,
            "metrics": self.node.metrics,
        }
        for peer in list(self.node.peers.keys()):
            if peer == self.node.node_id:
                continue  # skip self
            host, port = self.node.peers[peer]
            try:
                self.sock.sendto(json.dumps(msg).encode(), (host, port))
            except Exception as e:
                console.print(f"[ClusterManager] Failed to send heartbeat to {peer}: {e}")

    def listen(self):
        self.sock.bind((self.node.host, self.node.port))
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                msg = json.loads(data.decode())
                node_id = msg["node_id"]
                self.node.peers[node_id] = (msg["host"], msg["port"])
                # update metrics
                self.node.peers[node_id+"_metrics"] = msg["metrics"]
            except socket.timeout:
                continue
            except Exception as e:
                console.print(f"[ClusterManager] Error receiving data: {e}")

    def start(self):
        self.running = True
        Thread(target=self.listen, name="ClusterListener").start()
        Thread(target=self.heartbeat_loop, name="ClusterHeartbeat").start()

    def heartbeat_loop(self):
        while self.running:
            self.node.update_metrics()
            self.broadcast_heartbeat()
            time.sleep(self.sync_interval)

    def stop(self):
        self.running = False
        self.sock.close()

# ==============================
# 3. Predictive Scaler
# ==============================
class PredictiveScaler:
    def __init__(self, cluster_node):
        self.node = cluster_node
        self.threshold_cpu = 85
        self.threshold_ram = 90

    def evaluate_load(self):
        cpu = self.node.metrics["cpu"]
        ram = self.node.metrics["ram"]
        return cpu, ram

    def scale_services(self):
        cpu, ram = self.evaluate_load()
        if cpu > self.threshold_cpu or ram > self.threshold_ram:
            console.print(f"[PredictiveScaler] High load detected (CPU={cpu}%, RAM={ram}%), pausing non-critical services")
            for t in supervisor.threads:
                if getattr(t, "critical", False):
                    continue
                if t.is_alive():
                    console.print(f"[PredictiveScaler] Suggest pause for thread {t.name}")

predictive_scaler = PredictiveScaler(ClusterNode())

# ==============================
# 4. Distributed Agent
# ==============================
class DistributedAgent:
    def __init__(self, agent_id=None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.tasks = deque()
        self.running = True

    def assign_task(self, task):
        self.tasks.append(task)

    def run(self):
        while self.running:
            if self.tasks:
                task = self.tasks.popleft()
                try:
                    task()
                    console.print(f"[DistributedAgent] Agent {self.agent_id} completed a task")
                except Exception as e:
                    console.print(f"[DistributedAgent] Task failed: {e}")
            time.sleep(1)

distributed_agent = DistributedAgent()
Thread(target=distributed_agent.run, name="DistributedAgent").start()

# ==============================
# 5. Cluster Supervisor Integration
# ==============================
cluster_node = ClusterNode()
cluster_manager = ClusterManager(cluster_node)
supervisor.register_thread(Thread(target=cluster_manager.start, name="ClusterManager"))

console.print("[ONYX] Cluster sync, predictive scaling, and distributed agent integration active")

# ==============================
# PHASE 16: Network Security, IDS/IPS, VPN & Monitoring
# ==============================

import scapy.all as scapy
import subprocess
import queue
import ipaddress

# ==============================
# 1. Traffic Analyzer
# ==============================
class TrafficAnalyzer:
    def __init__(self, interface=None):
        self.interface = interface or "eth0"
        self.packet_queue = queue.Queue()
        self.active = True
        self.stats = defaultdict(lambda: {"count": 0, "bytes": 0})

    def packet_callback(self, pkt):
        try:
            src = pkt[scapy.IP].src
            dst = pkt[scapy.IP].dst
            length = len(pkt)
            self.stats[src]["count"] += 1
            self.stats[src]["bytes"] += length
        except Exception:
            pass

    def sniff_packets(self):
        while self.active:
            try:
                scapy.sniff(iface=self.interface, prn=self.packet_callback, store=False, timeout=5)
            except Exception as e:
                console.print(f"[TrafficAnalyzer] Sniffing error: {e}")

    def start(self):
        self.active = True
        Thread(target=self.sniff_packets, name="TrafficSniffer").start()

    def stop(self):
        self.active = False

# ==============================
# 2. Intrusion Detection System (IDS)
# ==============================
class IntrusionDetection:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.alerts = deque(maxlen=100)
        self.threshold = 1000  # packets per interval
        self.active = True

    def detect_anomalies(self):
        while self.active:
            for ip, stat in self.analyzer.stats.items():
                if stat["count"] > self.threshold:
                    alert = {"ip": ip, "count": stat["count"], "time": time.time()}
                    self.alerts.append(alert)
                    console.print(f"[IDS] Suspicious traffic from {ip}, count={stat['count']}")
            time.sleep(5)

    def start(self):
        self.active = True
        Thread(target=self.detect_anomalies, name="IDSMonitor").start()

    def stop(self):
        self.active = False

# ==============================
# 3. Firewall Manager
# ==============================
class FirewallManager:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule_args):
        self.rules.append(rule_args)
        try:
            subprocess.run(["ufw"] + rule_args, check=True)
            console.print(f"[Firewall] Applied rule: {rule_args}")
        except Exception as e:
            console.print(f"[Firewall] Failed to apply rule {rule_args}: {e}")

    def remove_rule(self, rule_args):
        try:
            subprocess.run(["ufw", "delete"] + rule_args, check=True)
            console.print(f"[Firewall] Removed rule: {rule_args}")
            self.rules.remove(rule_args)
        except Exception as e:
            console.print(f"[Firewall] Failed to remove rule {rule_args}: {e}")

# ==============================
# 4. VPN Manager (WireGuard)
# ==============================
class VPNManager:
    def __init__(self, config_path="/etc/wireguard/onyx.conf"):
        self.config_path = config_path
        self.active = False

    def connect(self):
        try:
            subprocess.run(["wg-quick", "up", self.config_path], check=True)
            self.active = True
            console.print("[VPNManager] VPN connected")
        except Exception as e:
            console.print(f"[VPNManager] Failed to connect VPN: {e}")
            self.active = False

    def disconnect(self):
        try:
            subprocess.run(["wg-quick", "down", self.config_path], check=True)
            self.active = False
            console.print("[VPNManager] VPN disconnected")
        except Exception as e:
            console.print(f"[VPNManager] Failed to disconnect VPN: {e}")

# ==============================
# 5. Network Monitor
# ==============================
class NetworkMonitor:
    def __init__(self):
        self.bytes_sent_prev, self.bytes_recv_prev = psutil.net_io_counters()[:2]
        self.active = True
        self.load_threshold = 80  # % utilization

    def monitor_loop(self):
        while self.active:
            io = psutil.net_io_counters()
            sent = io.bytes_sent - self.bytes_sent_prev
            recv = io.bytes_recv - self.bytes_recv_prev
            self.bytes_sent_prev, self.bytes_recv_prev = io.bytes_sent, io.bytes_recv
            utilization = (sent + recv) / 1e6  # MB per interval
            if utilization > self.load_threshold:
                console.print(f"[NetworkMonitor] High network usage: {utilization:.2f} MB")
            time.sleep(5)

    def start(self):
        self.active = True
        Thread(target=self.monitor_loop, name="NetworkMonitor").start()

    def stop(self):
        self.active = False

# ==============================
# 6. Integration into Supervisor
# ==============================
traffic_analyzer = TrafficAnalyzer()
ids = IntrusionDetection(traffic_analyzer)
firewall_manager = FirewallManager()
vpn_manager = VPNManager()
network_monitor = NetworkMonitor()

supervisor.register_thread(Thread(target=traffic_analyzer.start, name="TrafficAnalyzer"))
supervisor.register_thread(Thread(target=ids.start, name="IntrusionDetection"))
supervisor.register_thread(Thread(target=network_monitor.start, name="NetworkMonitor"))

console.print("[ONYX] Network security, IDS/IPS, VPN, and monitoring integrated")

# ==============================
# PHASE 17: API & Web Dashboard
# ==============================

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
import jwt

# ==============================
# 1. API Auth
# ==============================
class APIAuth:
    SECRET = "onyx_api_secret_key"  # In production, load from secure store
    ALGORITHM = "HS256"

    @classmethod
    def encode(cls, payload):
        return jwt.encode(payload, cls.SECRET, algorithm=cls.ALGORITHM)

    @classmethod
    def decode(cls, token):
        try:
            return jwt.decode(token, cls.SECRET, algorithms=[cls.ALGORITHM])
        except jwt.PyJWTError:
            raise HTTPException(status_code=403, detail="Invalid auth token")

bearer_scheme = HTTPBearer()

def auth_required(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    return APIAuth.decode(credentials.credentials)

# ==============================
# 2. FastAPI App
# ==============================
app = FastAPI(title="ONYX API", version="1.0")

@app.get("/api/v1/status")
def get_status(user=Depends(auth_required)):
    return {
        "system": {
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent,
            "uptime": time.time() - psutil.boot_time()
        },
        "services": [s.name for s in service_manager.services],
        "ai_metrics": ai_core.get_metrics(),
        "network": {"traffic": traffic_analyzer.stats, "vpn": vpn_manager.active}
    }

@app.get("/api/v1/backup")
def list_backups(user=Depends(auth_required)):
    backups = backup_manager.list_backups()
    return {"backups": backups}

@app.post("/api/v1/backup/run")
def run_backup(user=Depends(auth_required)):
    backup_manager.perform_backup()
    return {"status": "backup_started"}

@app.get("/api/v1/logs")
def get_logs(lines: int = 20, user=Depends(auth_required)):
    return {"logs": log_manager.get_recent(lines)}

# ==============================
# 3. Embedded Web Dashboard
# ==============================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ONYX Dashboard</title>
<style>
body { font-family: sans-serif; background: #111; color: #eee; }
h1 { color: #1abc9c; }
pre { background: #222; padding: 10px; }
</style>
<script>
async function fetchStatus() {
    const token = localStorage.getItem('onyx_token');
    const res = await fetch('/api/v1/status', {headers:{'Authorization':'Bearer '+token}});
    const data = await res.json();
    document.getElementById('status').textContent = JSON.stringify(data,null,2);
}
setInterval(fetchStatus, 3000);
</script>
</head>
<body>
<h1>ONYX Dashboard</h1>
<pre id="status">{Loading...}</pre>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML

# ==============================
# 4. API Server Runner
# ==============================
def start_api_server():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

supervisor.register_thread(Thread(target=start_api_server, name="FastAPI_Server"))

console.print("[ONYX] API & Web Dashboard integrated")

# ==============================
# PHASE 18: Multi-Agent AI & Reinforcement
# ==============================

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ==============================
# 1. Neural Core for Agents
# ==============================
class NeuralCore(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# ==============================
# 2. Reinforcement Learning Agent
# ==============================
class AIAgent(Thread):
    def __init__(self, name, state_size, action_size, memory_capacity=1000, gamma=0.99, lr=0.001):
        super().__init__(name=name)
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_capacity)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.model = NeuralCore(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.active = True
        self.reward_system = reward_system

    def run(self):
        while self.active:
            state = self.observe_state()
            action = self.choose_action(state)
            reward, next_state = self.take_action(action)
            self.memory.append((state, action, reward, next_state))
            self.learn()
            time.sleep(0.5)

    def observe_state(self):
        # Flatten system metrics into a state vector
        return torch.tensor([
            psutil.cpu_percent()/100.0,
            psutil.virtual_memory().percent/100.0,
            len(service_manager.services)/10.0,
            self.reward_system.get_total()/100.0
        ], dtype=torch.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def take_action(self, action):
        reward = 0
        # Action mapping: 0=optimize, 1=backup, 2=monitor, 3=game-play
        if action == 0:
            reward += optimizer.optimize_resources()
        elif action == 1:
            backup_manager.perform_backup()
            reward += 10
        elif action == 2:
            traffic_analyzer.scan_network()
            reward += 2
        elif action == 3:
            game_engine.play_turn()
            reward += 5
        next_state = self.observe_state()
        return reward, next_state

    def learn(self):
        if len(self.memory) < 32:
            return
        batch = random.sample(self.memory, 32)
        for state, action, reward, next_state in batch:
            target = reward + self.gamma * torch.max(self.model(next_state))
            current = self.model(state)[action]
            loss = self.loss_fn(current, target.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def terminate(self):
        self.active = False

# ==============================
# 3. Multi-Agent Manager
# ==============================
class AgentManager:
    def __init__(self, agent_count=5):
        self.agents = []
        for i in range(agent_count):
            agent = AIAgent(f"Agent-{i}", state_size=4, action_size=4)
            self.agents.append(agent)
            supervisor.register_thread(agent)

    def start_all(self):
        for agent in self.agents:
            agent.start()
        console.print(f"[ONYX] Started {len(self.agents)} AI agents")

    def stop_all(self):
        for agent in self.agents:
            agent.terminate()
            agent.join()
        console.print("[ONYX] All AI agents stopped")

# ==============================
# 4. Game Engine Hook (AI-driven)
# ==============================
class GameEngine:
    def play_turn(self):
        # Minimal AI for demonstration
        result = random.choice(["win", "lose", "draw"])
        reward_system.add_points(1 if result=="win" else 0)
        return result

game_engine = GameEngine()
agent_manager = AgentManager()
agent_manager.start_all()

console.print("[ONYX] Multi-Agent AI & RL layer online")

# ==============================
# PHASE 19: Distributed Agent Communication & Cluster Learning
# ==============================

import zmq
import json
import socket

# ==============================
# 1. Cluster Node Representation
# ==============================
class ClusterNode:
    def __init__(self, node_id=None, ip=None, port=5555):
        self.node_id = node_id or self.generate_node_id()
        self.ip = ip or self.get_local_ip()
        self.port = port
        self.metrics = {}
        self.last_heartbeat = time.time()

    def generate_node_id(self):
        # Persistent node ID based on MAC
        try:
            mac = open('/sys/class/net/eth0/address').read().strip()
            return f"node-{mac.replace(':','')}"
        except:
            return f"node-{uuid.uuid4().hex}"

    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except:
            return "127.0.0.1"
        finally:
            s.close()

# ==============================
# 2. Distributed Agent Manager
# ==============================
class DistributedAgentManager:
    def __init__(self, cluster_nodes=None):
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.sub_socket = self.context.socket(zmq.SUB)
        self.cluster_nodes = cluster_nodes or []
        self.active = True

    def start(self):
        self.pub_socket.bind("tcp://*:5555")
        for node in self.cluster_nodes:
            if node.ip != self.get_local_ip():
                self.sub_socket.connect(f"tcp://{node.ip}:{node.port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.thread = Thread(target=self.listen_loop, name="DistributedAgentListener")
        supervisor.register_thread(self.thread)
        self.thread.start()
        console.print("[ONYX] Distributed agent communication started")

    def listen_loop(self):
        while self.active:
            try:
                msg = self.sub_socket.recv(flags=zmq.NOBLOCK)
                data = json.loads(msg.decode())
                self.handle_message(data)
            except zmq.Again:
                time.sleep(0.1)

    def handle_message(self, data):
        # Merge remote agent experience
        if data.get("type") == "agent_experience":
            agent_id = data.get("agent_id")
            experience = data.get("experience")
            reward_system.add_points(experience.get("reward", 0))
            console.print(f"[ONYX] Received experience from {agent_id}")

    def broadcast_experience(self, agent_id, experience):
        msg = json.dumps({
            "type": "agent_experience",
            "agent_id": agent_id,
            "experience": experience
        }).encode()
        self.pub_socket.send(msg)

    def stop(self):
        self.active = False
        self.thread.join()
        self.pub_socket.close()
        self.sub_socket.close()
        self.context.term()
        console.print("[ONYX] Distributed agent manager stopped")

    def get_local_ip(self):
        return ClusterNode().get_local_ip()

# ==============================
# 3. Cluster Synchronization Loop
# ==============================
class ClusterSync:
    def __init__(self, nodes):
        self.nodes = nodes
        self.active = True

    def start(self):
        self.thread = Thread(target=self.sync_loop, name="ClusterSyncThread")
        supervisor.register_thread(self.thread)
        self.thread.start()
        console.print("[ONYX] Cluster synchronization loop started")

    def sync_loop(self):
        while self.active:
            for node in self.nodes:
                if node.ip != self.get_local_ip():
                    try:
                        heartbeat = {
                            "node_id": node.node_id,
                            "metrics": {
                                "cpu": psutil.cpu_percent(),
                                "ram": psutil.virtual_memory().percent,
                                "services": len(service_manager.services)
                            },
                            "timestamp": time.time()
                        }
                        self.send_heartbeat(node.ip, node.port, heartbeat)
                        node.last_heartbeat = time.time()
                    except Exception as e:
                        console.print(f"[ONYX] ClusterSync error: {e}")
            time.sleep(5)

    def send_heartbeat(self, ip, port, heartbeat):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(json.dumps(heartbeat).encode(), (ip, port))
        except Exception as e:
            console.print(f"[ONYX] Heartbeat failed: {e}")
        finally:
            s.close()

    def stop(self):
        self.active = False
        self.thread.join()
        console.print("[ONYX] ClusterSync stopped")

    def get_local_ip(self):
        return ClusterNode().get_local_ip()

# ==============================
# 4. Integration with AgentManager
# ==============================
distributed_manager = DistributedAgentManager(cluster_nodes=[ClusterNode()])
distributed_manager.start()

def broadcast_agent_experience(agent_id, experience):
    distributed_manager.broadcast_experience(agent_id, experience)

# Hook into AIAgent to broadcast after learning
original_learn = AIAgent.learn
def new_learn(self):
    original_learn(self)
    experience = {
        "reward": self.reward_system.get_total(),
        "state": self.observe_state().tolist()
    }
    broadcast_agent_experience(self.name, experience)
AIAgent.learn = new_learn

cluster_sync = ClusterSync([ClusterNode()])
cluster_sync.start()

console.print("[ONYX] Phase 19: Distributed agent communication & cluster learning online")

# ==============================
# PHASE 20: Hybrid Sandbox & Security Layer
# ==============================

import subprocess
import docker
import resource
import tempfile
import shutil

# ==============================
# 1. Hybrid Sandbox for Python, C++, Shell
# ==============================
class HybridSandbox:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.active = True

    # ------------------------------
    # Python code execution
    # ------------------------------
    def run_python(self, code, timeout=5):
        try:
            container = self.docker_client.containers.run(
                image="python:3.11-slim",
                command=["python", "-c", code],
                detach=False,
                remove=True,
                network_mode="none",
                mem_limit="256m",
                cpu_quota=50000  # 50% of one CPU
            )
        except docker.errors.ContainerError as e:
            console.print(f"[ONYX] Python sandbox error: {e}")
        except Exception as e:
            console.print(f"[ONYX] Unexpected Python sandbox exception: {e}")

    # ------------------------------
    # C++ code execution
    # ------------------------------
    def run_cpp(self, cpp_code, timeout=10):
        temp_dir = tempfile.mkdtemp()
        source_path = f"{temp_dir}/sandbox.cpp"
        binary_path = f"{temp_dir}/sandbox.out"

        with open(source_path, "w") as f:
            f.write(cpp_code)

        try:
            # Compile C++
            subprocess.run(["g++", source_path, "-o", binary_path],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Run compiled binary
            subprocess.run([binary_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            console.print(f"[ONYX] C++ sandbox error: {e.stderr.decode()}")
        finally:
            shutil.rmtree(temp_dir)

    # ------------------------------
    # Shell script execution
    # ------------------------------
    def run_shell(self, script_code, shell="/bin/bash", timeout=5):
        temp_dir = tempfile.mkdtemp()
        script_path = f"{temp_dir}/sandbox.sh"

        with open(script_path, "w") as f:
            f.write(script_code)
        os.chmod(script_path, 0o700)

        try:
            subprocess.run([shell, script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            console.print(f"[ONYX] Shell sandbox error: {e.stderr.decode()}")
        finally:
            shutil.rmtree(temp_dir)

# ==============================
# 2. Security / Permissions Layer
# ==============================
class SecurityManager:
    def __init__(self, secret_file=None):
        self.secret_file = secret_file or os.path.expanduser("~/.onyx/secret.json")
        self.authenticated = False
        self.admin_token = None
        self.load_secret()

    def load_secret(self):
        if os.path.exists(self.secret_file):
            with open(self.secret_file, "r") as f:
                try:
                    data = json.load(f)
                    self.admin_token = data.get("admin_token")
                except:
                    console.print("[ONYX] Secret file corrupted")
        else:
            self.admin_token = uuid.uuid4().hex
            os.makedirs(os.path.dirname(self.secret_file), exist_ok=True)
            with open(self.secret_file, "w") as f:
                json.dump({"admin_token": self.admin_token}, f)
                os.chmod(self.secret_file, 0o600)

    def authenticate(self, token=None):
        if token is None:
            # CLI prompt if interactive
            from prompt_toolkit import prompt
            token = prompt("ONYX Admin Token: ", is_password=True)
        if token == self.admin_token:
            self.authenticated = True
            console.print("[ONYX] Authentication successful")
        else:
            console.print("[ONYX] Authentication failed")
            self.authenticated = False
        return self.authenticated

    def is_authenticated(self):
        return self.authenticated

# ==============================
# 3. Resource Limits
# ==============================
def limit_resources(cpu_seconds=2, memory_mb=256):
    # CPU time
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    # Memory
    resource.setrlimit(resource.RLIMIT_AS, (memory_mb*1024*1024, memory_mb*1024*1024))

# ==============================
# 4. Sandbox Integration with Agents
# ==============================
sandbox = HybridSandbox()

# Hook agent tasks to sandbox execution for testing
original_process_task = AIAgent.process_task
def sandboxed_process_task(self):
    task = self.task_queue.get()
    code = task.get("code", "")
    language = task.get("language", "python")

    if language == "python":
        sandbox.run_python(code)
    elif language == "cpp":
        sandbox.run_cpp(code)
    elif language == "shell":
        sandbox.run_shell(code.get("script", ""))
    else:
        console.print(f"[ONYX] Unknown language {language} in task")

    original_process_task(self)

AIAgent.process_task = sandboxed_process_task

console.print("[ONYX] Phase 20: Hybrid sandbox & security layer online")

# ==============================
# PHASE 21: API & Remote Control Layer
# ==============================

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import time
import json
import jwt

# ==============================
# 1. API Authentication
# ==============================
SECRET_KEY = uuid.uuid4().hex
ALGORITHM = "HS256"

security = HTTPBearer()

def authenticate_api_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Insufficient privileges")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True

def generate_api_token(role="admin", expires_in=3600):
    payload = {"role": role, "exp": time.time() + expires_in}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# ==============================
# 2. FastAPI App Setup
# ==============================
app = FastAPI(
    title="ONYX Remote Control API",
    description="Secure REST API for ONYX autonomous system",
    version="0.9"
)

# Allow local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 3. System Status Endpoint
# ==============================
@app.get("/api/v1/status")
def api_status(auth: bool = Depends(authenticate_api_token)):
    status = {
        "system": "online" if onyx_core.is_active else "offline",
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "agents": len(onyx_core.agents),
        "tasks_pending": sum([len(agent.task_queue.queue) for agent in onyx_core.agents])
    }
    return status

# ==============================
# 4. AI & Task Management Endpoints
# ==============================
@app.get("/api/v1/ai_status")
def api_ai_status(auth: bool = Depends(authenticate_api_token)):
    return onyx_core.get_ai_metrics()

@app.post("/api/v1/submit_task")
def api_submit_task(task: dict, auth: bool = Depends(authenticate_api_token)):
    """
    Task structure:
    {
        "language": "python/cpp/shell",
        "code": "...",
        "priority": 1
    }
    """
    onyx_core.dispatch_task(task)
    return {"status": "accepted", "task_id": uuid.uuid4().hex}

@app.get("/api/v1/agents")
def api_agents(auth: bool = Depends(authenticate_api_token)):
    return [agent.id for agent in onyx_core.agents]

@app.post("/api/v1/stop_agent/{agent_id}")
def api_stop_agent(agent_id: str, auth: bool = Depends(authenticate_api_token)):
    agent = next((a for a in onyx_core.agents if a.id == agent_id), None)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent.terminate()
    return {"status": "terminated", "agent_id": agent_id}

# ==============================
# 5. Metrics Reporting
# ==============================
@app.get("/api/v1/metrics")
def api_metrics(auth: bool = Depends(authenticate_api_token)):
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "network_io": psutil.net_io_counters()._asdict(),
        "disk_usage": psutil.disk_usage("/")._asdict(),
        "uptime": time.time() - psutil.boot_time(),
        "reward_points": reward_system.total
    }
    return metrics

# ==============================
# 6. Remote Control Functions
# ==============================
@app.post("/api/v1/trigger_backup")
def api_trigger_backup(auth: bool = Depends(authenticate_api_token)):
    backup_manager.perform_backup()
    return {"status": "backup_started"}

@app.post("/api/v1/restart_system")
def api_restart_system(auth: bool = Depends(authenticate_api_token)):
    console.print("[ONYX] Remote API requested system restart")
    threading.Thread(target=onyx_core.restart_system).start()
    return {"status": "restart_initiated"}

# ==============================
# 7. API Server Bootstrap
# ==============================
def run_api_server(host="0.0.0.0", port=8000):
    console.print(f"[ONYX] API server starting on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")

# ==============================
# 8. Background API Thread
# ==============================
api_thread = threading.Thread(target=run_api_server, daemon=True)
api_thread.start()

console.print("[ONYX] Phase 21: API & Remote Control Layer online")

# ==============================
# PHASE 22: Cluster, Consensus & Distributed Agents
# ==============================

import socket
import selectors
import struct
import threading
import pickle
import random

# ==============================
# 1. Cluster Node Representation
# ==============================
class ClusterNode:
    def __init__(self, host: str, port: int, node_id: str = None):
        self.host = host
        self.port = port
        self.id = node_id or self.generate_node_id()
        self.metrics = {}
        self.last_heartbeat = 0.0
        self.active = True

    def generate_node_id(self):
        # Persistent node ID using MAC address fallback
        try:
            mac = ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff) 
                            for i in range(0, 8*6, 8)][::-1])
            return hashlib.sha256(mac.encode()).hexdigest()[:12]
        except:
            return uuid.uuid4().hex[:12]

    def update_metrics(self):
        self.metrics = {
            "cpu": psutil.cpu_percent(),
            "ram": psutil.virtual_memory().percent,
            "tasks_pending": sum([len(agent.task_queue.queue) for agent in onyx_core.agents])
        }
        self.last_heartbeat = time.time()

# ==============================
# 2. Cluster Manager
# ==============================
class ClusterManager:
    def __init__(self, port=5050):
        self.nodes = {}  # node_id: ClusterNode
        self.local_node = ClusterNode(host="127.0.0.1", port=port)
        self.nodes[self.local_node.id] = self.local_node
        self.port = port
        self.selector = selectors.DefaultSelector()
        self.active = True
        self.lock = threading.Lock()
        console.print(f"[ONYX] Cluster Manager initialized at {self.local_node.host}:{port}")

    # ==============================
    # Heartbeat broadcasting
    # ==============================
    def broadcast_heartbeat(self):
        while self.active:
            self.local_node.update_metrics()
            heartbeat_data = pickle.dumps(self.local_node)
            with self.lock:
                for node_id, node in self.nodes.items():
                    if node_id == self.local_node.id:
                        continue
                    try:
                        with socket.create_connection((node.host, node.port), timeout=2) as sock:
                            sock.sendall(heartbeat_data)
                    except Exception as e:
                        console.print(f"[ClusterManager] Failed heartbeat to {node.host}:{node.port} - {e}")
            time.sleep(10)

    # ==============================
    # Heartbeat listener
    # ==============================
    def start_listener(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.local_node.host, self.port))
        listener.listen()
        listener.setblocking(False)
        self.selector.register(listener, selectors.EVENT_READ, self.accept)
        console.print(f"[ClusterManager] Listening for cluster nodes on {self.local_node.host}:{self.port}")

        while self.active:
            events = self.selector.select(timeout=1)
            for key, mask in events:
                callback = key.data
                callback(key.fileobj)

    def accept(self, sock):
        conn, addr = sock.accept()
        with conn:
            try:
                data = conn.recv(4096)
                node = pickle.loads(data)
                if isinstance(node, ClusterNode):
                    with self.lock:
                        self.nodes[node.id] = node
                    console.print(f"[ClusterManager] Received heartbeat from {node.id}")
            except Exception as e:
                console.print(f"[ClusterManager] Failed to process heartbeat: {e}")

    # ==============================
    # Leader Election (simplified)
    # ==============================
    def elect_leader(self):
        # Leader = node with lowest node_id lexicographically
        while self.active:
            with self.lock:
                if not self.nodes:
                    continue
                leader_id = min(self.nodes.keys())
                self.leader_id = leader_id
            time.sleep(15)

    # ==============================
    # Distributed Task Dispatch
    # ==============================
    def dispatch_task_to_node(self, task: dict):
        # Round-robin for simplicity
        active_nodes = [n for n in self.nodes.values() if n.active]
        if not active_nodes:
            console.print("[ClusterManager] No active nodes to dispatch task")
            return
        chosen_node = random.choice(active_nodes)
        if chosen_node.id == self.local_node.id:
            onyx_core.dispatch_task(task)
        else:
            try:
                with socket.create_connection((chosen_node.host, chosen_node.port), timeout=2) as sock:
                    sock.sendall(pickle.dumps(task))
                console.print(f"[ClusterManager] Task dispatched to node {chosen_node.id}")
            except Exception as e:
                console.print(f"[ClusterManager] Failed to send task to {chosen_node.id}: {e}")

# ==============================
# 3. Cluster Manager Initialization
# ==============================
cluster_manager = ClusterManager()
threading.Thread(target=cluster_manager.start_listener, daemon=True).start()
threading.Thread(target=cluster_manager.broadcast_heartbeat, daemon=True).start()
threading.Thread(target=cluster_manager.elect_leader, daemon=True).start()

console.print("[ONYX] Phase 22: Cluster & Distributed Agent Layer online")

# ==============================
# PHASE 23: Logging, Monitoring & Observability
# ==============================

import logging
from logging.handlers import TimedRotatingFileHandler
import json
import atexit
from collections import deque
from prometheus_client import start_http_server, Gauge, Counter

# ==============================
# 1. Log Manager
# ==============================
class LogManager:
    def __init__(self, log_file="onyx.log"):
        self.logger = logging.getLogger("ONYX")
        self.logger.setLevel(logging.INFO)
        self.log_file = log_file

        # Rotating handler: one file per day, keep last 30 days
        handler = TimedRotatingFileHandler(self.log_file, when="midnight", backupCount=30)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.console = console

    def info(self, msg, **kwargs):
        self.logger.info(msg)
        self.console.print(f"[INFO] {msg} {kwargs}")

    def warning(self, msg, **kwargs):
        self.logger.warning(msg)
        self.console.print(f"[WARN] {msg} {kwargs}")

    def error(self, msg, **kwargs):
        self.logger.error(msg)
        self.console.print(f"[ERROR] {msg} {kwargs}")

    def get_recent(self, n=20):
        # Return last n lines of log file
        lines = deque(maxlen=n)
        try:
            with open(self.log_file, "r") as f:
                for line in f:
                    lines.append(line.strip())
        except FileNotFoundError:
            pass
        return list(lines)

log_manager = LogManager()

# ==============================
# 2. Metrics Collection (Prometheus)
# ==============================
# Gauge metrics for real-time values
cpu_gauge = Gauge('onyx_cpu_percent', 'CPU Usage Percent')
ram_gauge = Gauge('onyx_ram_percent', 'RAM Usage Percent')
task_queue_gauge = Gauge('onyx_task_queue_length', 'Number of Tasks Pending')
backup_counter = Counter('onyx_backup_total', 'Total Backups Performed')
agent_reward_gauge = Gauge('onyx_agent_reward', 'Reward Points for Agents')

def collect_system_metrics():
    while True:
        cpu_gauge.set(psutil.cpu_percent())
        ram_gauge.set(psutil.virtual_memory().percent)
        total_tasks = sum([len(agent.task_queue.queue) for agent in onyx_core.agents])
        task_queue_gauge.set(total_tasks)
        time.sleep(5)

# ==============================
# 3. Event Monitoring
# ==============================
class EventMonitor:
    def __init__(self):
        self.alerts = deque(maxlen=500)

    def log_event(self, event_type, message, metadata=None):
        event = {
            "type": event_type,
            "message": message,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        self.alerts.append(event)
        log_manager.info(f"Event: {event_type} - {message}")

    def get_recent_alerts(self, n=20):
        return list(self.alerts)[-n:]

event_monitor = EventMonitor()

# ==============================
# 4. Audit Logger
# ==============================
class AuditLogger:
    def __init__(self, audit_file="onyx_audit.log"):
        self.audit_file = audit_file

    def audit(self, user, command, result, extra=None):
        record = {
            "timestamp": time.time(),
            "user": user,
            "command": command,
            "result": result,
            "extra": extra or {}
        }
        try:
            with open(self.audit_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            log_manager.error(f"Audit write failed: {e}")

audit_logger = AuditLogger()

# ==============================
# 5. Observability Thread Initialization
# ==============================
def start_observability():
    # Start Prometheus HTTP server on port 8001
    start_http_server(8001)
    # Start system metric collection
    threading.Thread(target=collect_system_metrics, daemon=True).start()
    log_manager.info("Observability and monitoring started (Prometheus metrics on port 8001)")

# Register atexit cleanup
atexit.register(lambda: log_manager.info("ONYX shutting down observability"))

start_observability()
console.print("[ONYX] Phase 23: Logging, Monitoring & Observability online")

# ==============================
# PHASE 24: API & Remote Control
# ==============================

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import jwt
import os

# ==============================
# 1. Configuration & Secrets
# ==============================
API_SECRET = os.environ.get("ONYX_API_SECRET", "supersecretkey")  # Use env var in production
API_TOKEN_EXPIRATION = 3600  # seconds

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, API_SECRET, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=403, detail="Invalid token")

# ==============================
# 2. API Models
# ==============================
class BackupRequest(BaseModel):
    paths: list[str]
    retention_days: int = 7

class ServiceCommand(BaseModel):
    service_name: str
    action: str  # start, stop, restart

# ==============================
# 3. FastAPI App
# ==============================
app = FastAPI(title="ONYX API", version="1.0")

@app.get("/api/v1/status")
def get_status(token: dict = Depends(verify_token)):
    """Return overall ONYX system status"""
    status = {
        "uptime": time.time() - onyx_core.start_time,
        "cpu_percent": psutil.cpu_percent(),
        "ram_percent": psutil.virtual_memory().percent,
        "agents": len(onyx_core.agents),
        "tasks_pending": sum([len(a.task_queue.queue) for a in onyx_core.agents]),
        "backups_done": backup_counter._value.get(),  # Prometheus Counter value
    }
    return status

@app.post("/api/v1/backup")
def trigger_backup(request: BackupRequest, token: dict = Depends(verify_token)):
    """Trigger a manual backup"""
    try:
        backup_manager.perform_backup(paths=request.paths, retention_days=request.retention_days)
        backup_counter.inc()
        audit_logger.audit(user=token.get("user", "api_user"), command="backup", result="success", extra={"paths": request.paths})
        return {"status": "Backup triggered", "paths": request.paths}
    except Exception as e:
        log_manager.error(f"Backup API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/service")
def control_service(request: ServiceCommand, token: dict = Depends(verify_token)):
    """Start, stop, or restart a service"""
    service = service_manager.get_service(request.service_name)
    if not service:
        raise HTTPException(status_code=404, detail=f"Service {request.service_name} not found")
    try:
        if request.action == "start":
            service.start()
        elif request.action == "stop":
            service.stop()
        elif request.action == "restart":
            service.restart()
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
        audit_logger.audit(user=token.get("user", "api_user"), command="service", result="success",
                           extra={"service": request.service_name, "action": request.action})
        return {"status": f"{request.action} executed for {request.service_name}"}
    except Exception as e:
        log_manager.error(f"Service control failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents")
def list_agents(token: dict = Depends(verify_token)):
    """Return status of all agents"""
    agents_status = []
    for agent in onyx_core.agents:
        agents_status.append({
            "id": agent.agent_id,
            "tasks_pending": len(agent.task_queue.queue),
            "active": agent.active
        })
    return agents_status

@app.get("/api/v1/logs")
def get_logs(n: int = 20, token: dict = Depends(verify_token)):
    """Return recent log lines"""
    return {"recent_logs": log_manager.get_recent(n)}

# ==============================
# 4. Token Generation Endpoint (for testing/automation)
# ==============================
@app.post("/api/v1/token")
def generate_token(user: str):
    """Generate a JWT token (for demo purposes, no password check here)"""
    payload = {
        "user": user,
        "exp": time.time() + API_TOKEN_EXPIRATION
    }
    token = jwt.encode(payload, API_SECRET, algorithm="HS256")
    return {"token": token}

# ==============================
# 5. API Runner
# ==============================
def start_api():
    log_manager.info("Starting ONYX API server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

threading.Thread(target=start_api, daemon=True).start()
console.print("[ONYX] Phase 24: API & Remote Control online (FastAPI on port 8000)")

# ==============================
# PHASE 25: Web UI & Dashboard
# ==============================

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Template

# ==============================
# 1. Simple In-Memory Template Engine
# ==============================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ONYX Dashboard</title>
<style>
body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 0; padding: 0; }
header { background: #222; padding: 1rem; text-align: center; font-size: 1.5rem; }
section { padding: 1rem; }
table { width: 100%; border-collapse: collapse; }
th, td { border: 1px solid #444; padding: 0.5rem; text-align: left; }
th { background: #333; }
tr:nth-child(even) { background: #1a1a1a; }
</style>
<script>
async function refreshStatus() {
    const res = await fetch('/api/v1/status', { headers: { 'Authorization': 'Bearer {{token}}' } });
    const data = await res.json();
    document.getElementById('uptime').innerText = data.uptime.toFixed(1) + "s";
    document.getElementById('cpu').innerText = data.cpu_percent + "%";
    document.getElementById('ram').innerText = data.ram_percent + "%";
}
setInterval(refreshStatus, 3000);
window.onload = refreshStatus;
</script>
</head>
<body>
<header>ONYX Dashboard</header>
<section>
<h2>System Metrics</h2>
<p>Uptime: <span id="uptime"></span></p>
<p>CPU Usage: <span id="cpu"></span></p>
<p>RAM Usage: <span id="ram"></span></p>
</section>
<section>
<h2>Agents</h2>
<table>
<tr><th>ID</th><th>Tasks Pending</th><th>Active</th></tr>
{% for agent in agents %}
<tr>
<td>{{agent.id}}</td>
<td>{{agent.tasks_pending}}</td>
<td>{{agent.active}}</td>
</tr>
{% endfor %}
</table>
</section>
<section>
<h2>Recent Logs</h2>
<pre>{{logs}}</pre>
</section>
</body>
</html>
"""

# ==============================
# 2. Dashboard Endpoint
# ==============================
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(token: dict = Depends(verify_token)):
    """Render ONYX Web Dashboard"""
    agents_status = [
        {"id": a.agent_id, "tasks_pending": len(a.task_queue.queue), "active": a.active}
        for a in onyx_core.agents
    ]
    recent_logs = "\n".join(log_manager.get_recent(30))
    template = Template(HTML_TEMPLATE)
    html = template.render(token="demo_token", agents=agents_status, logs=recent_logs)
    return HTMLResponse(content=html)

# ==============================
# 3. Static File Mount (Optional JS/CSS)
# ==============================
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==============================
# 4. Dashboard Runner
# ==============================
def start_dashboard():
    log_manager.info("Starting ONYX Web Dashboard")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

threading.Thread(target=start_dashboard, daemon=True).start()
console.print("[ONYX] Phase 25: Web UI & Dashboard online (http://0.0.0.0:8080/dashboard)")

# ==============================
# PHASE 26: Cluster Orchestration & Consensus
# ==============================

import socket
from pysyncobj import SyncObj, SyncObjConf, replicated
import json

# ==============================
# 1. Cluster Node Representation
# ==============================
class ClusterNode:
    def __init__(self, host, port, node_id=None):
        self.host = host
        self.port = port
        self.node_id = node_id or self.generate_node_id()
        self.metrics = {"cpu": 0.0, "ram": 0.0, "uptime": 0.0}
        self.last_heartbeat = time.time()

    def generate_node_id(self):
        # Persistent ID derived from host:port or fallback to UUID
        try:
            return hashlib.sha256(f"{self.host}:{self.port}".encode()).hexdigest()
        except Exception:
            return str(uuid.uuid4())

    def update_metrics(self):
        self.metrics["cpu"] = psutil.cpu_percent()
        self.metrics["ram"] = psutil.virtual_memory().percent
        self.metrics["uptime"] = time.time() - psutil.boot_time()
        self.last_heartbeat = time.time()

# ==============================
# 2. Cluster Manager
# ==============================
class ClusterManager(SyncObj):
    def __init__(self, self_addr, partner_addrs=None):
        cfg = SyncObjConf(dynamicMembershipChange=True)
        partner_addrs = partner_addrs or []
        super().__init__(self_addr, partner_addrs, conf=cfg)
        self.nodes = {}  # node_id: ClusterNode
        self.lock = threading.Lock()

    @replicated
    def register_node(self, node_info):
        node_id = node_info["node_id"]
        with self.lock:
            self.nodes[node_id] = ClusterNode(node_info["host"], node_info["port"], node_id)
        return node_id

    @replicated
    def remove_node(self, node_id):
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]

    def update_local_metrics(self):
        # Update own metrics
        local_node = ClusterNode("127.0.0.1", 5000)
        local_node.update_metrics()
        self.register_node({
            "node_id": local_node.node_id,
            "host": local_node.host,
            "port": local_node.port,
        })

    def choose_best_node(self):
        with self.lock:
            sorted_nodes = sorted(
                self.nodes.values(),
                key=lambda n: n.metrics.get("cpu", float('inf'))
            )
            return sorted_nodes[0] if sorted_nodes else None

    def heartbeat_loop(self, interval=5):
        while True:
            try:
                self.update_local_metrics()
            except Exception as e:
                log_manager.error(f"[Cluster] Heartbeat error: {e}")
            time.sleep(interval)

# ==============================
# 3. Distributed Task Scheduler
# ==============================
class DistributedScheduler:
    def __init__(self, cluster: ClusterManager):
        self.cluster = cluster
        self.task_queue = queue.Queue()
        self.active = True

    def submit_task(self, task):
        self.task_queue.put(task)

    def distribute_tasks(self):
        while self.active:
            try:
                task = self.task_queue.get(timeout=1)
                best_node = self.cluster.choose_best_node()
                if best_node:
                    log_manager.info(f"[Cluster] Assigning task {task['name']} to {best_node.node_id}")
                    # In real system: send via RPC / gRPC
                    # Here we simulate by adding locally
                    onyx_core.agents[0].task_queue.put(task)
                else:
                    log_manager.warning("[Cluster] No available node, retrying task later")
                    self.task_queue.put(task)
            except queue.Empty:
                continue
            except Exception as e:
                log_manager.error(f"[Cluster] Task distribution error: {e}")
            time.sleep(0.1)

# ==============================
# 4. Phase 26 Bootstrap
# ==============================
cluster_manager = ClusterManager("127.0.0.1:5000", [])
threading.Thread(target=cluster_manager.heartbeat_loop, daemon=True).start()

distributed_scheduler = DistributedScheduler(cluster_manager)
threading.Thread(target=distributed_scheduler.distribute_tasks, daemon=True).start()

console.print("[ONYX] Phase 26: Cluster Orchestration & Consensus online")

# ==============================
# PHASE 27: Advanced AI Reinforcement & Decision Loops
# ==============================

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ==============================
# 1. Neural Core for Decision Making
# ==============================
class DecisionNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# ==============================
# 2. Reinforcement Agent
# ==============================
class ReinforcementAgent:
    def __init__(self, state_size=10, action_size=5, lr=0.001, gamma=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 0.1
        self.memory = deque(maxlen=10000)
        self.model = DecisionNN(input_size=state_size, output_size=action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            probs = self.model(state_tensor)
        return torch.argmax(probs).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)

        q_values = self.model(states_tensor)
        next_q_values = self.model(next_states_tensor).detach()
        target = q_values.clone()

        for i in range(batch_size):
            target[i, actions_tensor[i]] = rewards_tensor[i] + self.gamma * torch.max(next_q_values[i]) * (1 - dones_tensor[i])

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ==============================
# 3. Multi-Agent Reinforcement Loop
# ==============================
class MultiAgentManager:
    def __init__(self, num_agents=3, state_size=10, action_size=5):
        self.agents = [ReinforcementAgent(state_size, action_size) for _ in range(num_agents)]
        self.active = True

    def simulate_environment(self, agent_id):
        # Example: generate state vector from system metrics
        cpu = psutil.cpu_percent() / 100.0
        ram = psutil.virtual_memory().percent / 100.0
        network_load = random.random()
        disk_io = random.random()
        cluster_load = len(cluster_manager.nodes) / 10.0
        state = [cpu, ram, network_load, disk_io, cluster_load] + [0]*(10-5)
        return state

    def run_loop(self):
        while self.active:
            for idx, agent in enumerate(self.agents):
                state = self.simulate_environment(idx)
                action = agent.select_action(state)

                # Map action to system operations
                reward = self.execute_action(idx, action)
                next_state = self.simulate_environment(idx)
                done = False  # continuous loop

                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step(batch_size=32)
            time.sleep(1)

    def execute_action(self, agent_id, action):
        # Map agent actions to cluster/service decisions
        reward = 0
        try:
            if action == 0:
                # Optimize CPU usage
                node = cluster_manager.choose_best_node()
                if node: reward += 1.0
            elif action == 1:
                # Trigger backup
                backup_manager.perform_backup()
                reward += 2.0
            elif action == 2:
                # Scale a service
                scheduler.scale_service("generic_service")
                reward += 1.5
            elif action == 3:
                # Redistribute tasks
                distributed_scheduler.distribute_tasks()
                reward += 1.0
            elif action == 4:
                # Self-heal infra
                infra_healer.run_checks()
                reward += 1.5
        except Exception as e:
            log_manager.error(f"[AI] Agent {agent_id} action failed: {e}")
            reward -= 1.0
        return reward

# ==============================
# 4. Phase 27 Bootstrap
# ==============================
multi_agent_manager = MultiAgentManager(num_agents=5)
threading.Thread(target=multi_agent_manager.run_loop, daemon=True).start()

console.print("[ONYX] Phase 27: Advanced AI Reinforcement Loop online")

# ==============================
# PHASE 28: Predictive Scaling & Proactive Resource Management
# ==============================

import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# ==============================
# 1. Predictive Resource Model
# ==============================
class ResourcePredictor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.cpu_history = deque(maxlen=window_size)
        self.ram_history = deque(maxlen=window_size)
        self.model_cpu = LinearRegression()
        self.model_ram = LinearRegression()
        self.trained = False

    def add_sample(self, cpu_percent, ram_percent):
        timestamp = len(self.cpu_history)
        self.cpu_history.append((timestamp, cpu_percent))
        self.ram_history.append((timestamp, ram_percent))

    def predict_next(self):
        if len(self.cpu_history) < self.window_size:
            return None, None
        X = np.array([i for i,_ in self.cpu_history]).reshape(-1,1)
        y_cpu = np.array([v for _,v in self.cpu_history])
        y_ram = np.array([v for _,v in self.ram_history])
        self.model_cpu.fit(X, y_cpu)
        self.model_ram.fit(X, y_ram)
        next_step = np.array([[len(self.cpu_history)]])
        pred_cpu = self.model_cpu.predict(next_step)[0]
        pred_ram = self.model_ram.predict(next_step)[0]
        return pred_cpu, pred_ram

# ==============================
# 2. Proactive Scaling Manager
# ==============================
class PredictiveScaler:
    def __init__(self, cpu_threshold=80, ram_threshold=85):
        self.predictor = ResourcePredictor(window_size=15)
        self.cpu_threshold = cpu_threshold
        self.ram_threshold = ram_threshold
        self.active = True

    def sample_metrics(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.predictor.add_sample(cpu, ram)
        return cpu, ram

    def scale_if_needed(self):
        pred_cpu, pred_ram = self.predictor.predict_next()
        if pred_cpu is None:
            return
        # Scaling decision
        if pred_cpu > self.cpu_threshold:
            # Identify services to scale up
            service = scheduler.select_scalable_service("cpu_intensive")
            if service:
                scheduler.scale_service(service, up=True)
                log_manager.info(f"[PredictiveScaler] CPU predicted high ({pred_cpu:.1f}%). Scaled {service}.")
        if pred_ram > self.ram_threshold:
            service = scheduler.select_scalable_service("memory_intensive")
            if service:
                scheduler.scale_service(service, up=True)
                log_manager.info(f"[PredictiveScaler] RAM predicted high ({pred_ram:.1f}%). Scaled {service}.")

    def run_loop(self):
        while self.active:
            self.sample_metrics()
            self.scale_if_needed()
            time.sleep(2)

# ==============================
# 3. Integration with Multi-Agent AI
# ==============================
class AIResourceIntegrator:
    def __init__(self, scaler, multi_agent_manager):
        self.scaler = scaler
        self.agents = multi_agent_manager.agents
        self.active = True

    def evaluate_agent_recommendations(self):
        for idx, agent in enumerate(self.agents):
            state = multi_agent_manager.simulate_environment(idx)
            action = agent.select_action(state)
            if action == 2:  # scale a service
                self.scaler.scale_if_needed()

    def run_loop(self):
        while self.active:
            self.evaluate_agent_recommendations()
            time.sleep(1)

# ==============================
# 4. Phase 28 Bootstrap
# ==============================
predictive_scaler = PredictiveScaler(cpu_threshold=75, ram_threshold=80)
threading.Thread(target=predictive_scaler.run_loop, daemon=True).start()

ai_integrator = AIResourceIntegrator(predictive_scaler, multi_agent_manager)
threading.Thread(target=ai_integrator.run_loop, daemon=True).start()

console.print("[ONYX] Phase 28: Predictive Scaling & Proactive Resource Management online")

# ==============================
# PHASE 29: Cluster-Wide Predictive Load Balancing & Intelligent Failover
# ==============================

import zmq
import json
from datetime import datetime

# ==============================
# 1. Cluster Node Representation
# ==============================
class ClusterNode:
    def __init__(self, node_id, ip, port):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.metrics = {"cpu": 0, "ram": 0, "services": {}}
        self.last_seen = datetime.utcnow()

    def update_metrics(self, cpu, ram, service_metrics):
        self.metrics["cpu"] = cpu
        self.metrics["ram"] = ram
        self.metrics["services"] = service_metrics
        self.last_seen = datetime.utcnow()

# ==============================
# 2. Cluster Manager
# ==============================
class ClusterManager:
    def __init__(self, node_id, ip="127.0.0.1", port=5555):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.nodes = {node_id: ClusterNode(node_id, ip, port)}
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{port}")
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.active = True

    def register_node(self, node_ip, node_port):
        self.sub_socket.connect(f"tcp://{node_ip}:{node_port}")

    def broadcast_metrics(self, metrics):
        message = json.dumps({"node_id": self.node_id, "metrics": metrics})
        self.pub_socket.send_string(message)

    def receive_metrics(self):
        try:
            msg = self.sub_socket.recv_string(flags=zmq.NOBLOCK)
            data = json.loads(msg)
            node_id = data["node_id"]
            metrics = data["metrics"]
            if node_id not in self.nodes:
                self.nodes[node_id] = ClusterNode(node_id, "unknown", 0)
            self.nodes[node_id].update_metrics(metrics["cpu"], metrics["ram"], metrics["services"])
        except zmq.Again:
            pass

    def get_least_loaded_node(self, exclude_ids=[]):
        min_load = float('inf')
        target_node = None
        for node_id, node in self.nodes.items():
            if node_id in exclude_ids:
                continue
            load = node.metrics["cpu"] + node.metrics["ram"]
            if load < min_load:
                min_load = load
                target_node = node
        return target_node

# ==============================
# 3. Intelligent Load Balancer
# ==============================
class PredictiveClusterBalancer:
    def __init__(self, cluster_manager, scaler):
        self.cluster_manager = cluster_manager
        self.scaler = scaler
        self.active = True

    def evaluate_and_redistribute(self):
        # Collect predictions from each node
        node_predictions = {}
        for node_id, node in self.cluster_manager.nodes.items():
            pred_cpu, pred_ram = self.scaler.predictor.predict_next()
            if pred_cpu is not None:
                node_predictions[node_id] = pred_cpu + pred_ram

        # Identify overloaded nodes
        overloaded_nodes = [nid for nid, load in node_predictions.items() if load > 150]

        for nid in overloaded_nodes:
            node = self.cluster_manager.nodes[nid]
            # Identify services to move
            for svc_name, svc_metrics in node.metrics["services"].items():
                target_node = self.cluster_manager.get_least_loaded_node(exclude_ids=[nid])
                if target_node:
                    # Move service
                    scheduler.migrate_service(svc_name, source=node.node_id, destination=target_node.node_id)
                    log_manager.info(f"[ClusterBalancer] Migrated {svc_name} from {node.node_id} to {target_node.node_id}")

    def run_loop(self):
        while self.active:
            self.cluster_manager.receive_metrics()
            self.evaluate_and_redistribute()
            # Broadcast local metrics
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            svc_metrics = scheduler.get_service_metrics()
            self.cluster_manager.broadcast_metrics({"cpu": cpu, "ram": ram, "services": svc_metrics})
            time.sleep(3)

# ==============================
# 4. Phase 29 Bootstrap
# ==============================
cluster_manager = ClusterManager(node_id=onyx_core.node_id)
# Example: connect to other nodes (replace IPs with actual cluster nodes)
# cluster_manager.register_node("192.168.1.10", 5555)
# cluster_manager.register_node("192.168.1.11", 5555)

cluster_balancer = PredictiveClusterBalancer(cluster_manager, predictive_scaler)
threading.Thread(target=cluster_balancer.run_loop, daemon=True).start()

console.print("[ONYX] Phase 29: Cluster-Wide Predictive Load Balancing & Intelligent Failover online")

# ==============================
# PHASE 30: Full Cluster Consensus & Autonomous Decision Making
# ==============================

import random
from pysyncobj import SyncObj, SyncObjConf, replicated

# ==============================
# 1. Cluster Consensus Node
# ==============================
class ClusterConsensusNode(SyncObj):
    def __init__(self, self_addr, partner_addrs):
        cfg = SyncObjConf(dynamicMembershipChange=True, fullDumpFile=f'/tmp/onyx_sync_{onyx_core.node_id}.dump')
        super().__init__(self_addr, partner_addrs, cfg)
        self.global_decisions = []

    @replicated
    def propose_decision(self, decision):
        self.global_decisions.append({"decision": decision, "timestamp": datetime.utcnow().isoformat()})
        log_manager.info(f"[Consensus] Node {onyx_core.node_id} proposed decision: {decision}")

    def get_latest_decision(self):
        if self.global_decisions:
            return self.global_decisions[-1]
        return None

# ==============================
# 2. Multi-Agent Global Evaluator
# ==============================
class GlobalDecisionEngine:
    def __init__(self, cluster_consensus_node):
        self.consensus_node = cluster_consensus_node
        self.active = True

    def evaluate_cluster_state(self):
        # Aggregate metrics across all nodes
        node_metrics = cluster_manager.nodes
        decisions = []

        for node_id, node in node_metrics.items():
            cpu = node.metrics["cpu"]
            ram = node.metrics["ram"]
            if cpu > 85 or ram > 85:
                decisions.append({"action": "migrate", "node": node_id})
            elif cpu < 30 and ram < 30:
                decisions.append({"action": "scale_up", "node": node_id})

        # Random exploration for RL
        if random.random() < 0.05:
            decisions.append({"action": "experimental", "node": random.choice(list(node_metrics.keys()))})

        return decisions

    def propose_cluster_decision(self):
        decisions = self.evaluate_cluster_state()
        for decision in decisions:
            self.consensus_node.propose_decision(decision)

# ==============================
# 3. Autonomous Cluster Executor
# ==============================
class AutonomousExecutor:
    def __init__(self, cluster_manager, decision_engine):
        self.cluster_manager = cluster_manager
        self.decision_engine = decision_engine
        self.active = True

    def execute_decisions(self):
        latest_decision = self.decision_engine.consensus_node.get_latest_decision()
        if latest_decision:
            decision = latest_decision["decision"]
            if decision["action"] == "migrate":
                svc_to_move = scheduler.get_high_load_service(decision["node"])
                target_node = self.cluster_manager.get_least_loaded_node(exclude_ids=[decision["node"]])
                if target_node and svc_to_move:
                    scheduler.migrate_service(svc_to_move, source=decision["node"], destination=target_node.node_id)
                    log_manager.info(f"[Executor] Migrated {svc_to_move} from {decision['node']} to {target_node.node_id}")
            elif decision["action"] == "scale_up":
                scheduler.scale_node_services(decision["node"])
                log_manager.info(f"[Executor] Scaled up services on node {decision['node']}")
            elif decision["action"] == "experimental":
                scheduler.trigger_experimental_task(decision["node"])
                log_manager.info(f"[Executor] Triggered experimental task on node {decision['node']}")

    def run_loop(self):
        while self.active:
            self.decision_engine.propose_cluster_decision()
            self.execute_decisions()
            time.sleep(5)

# ==============================
# 4. Phase 30 Bootstrap
# ==============================
# Example: local node and partner nodes
self_addr = f"127.0.0.1:{6000 + int(onyx_core.node_id[:4], 16) % 1000}"
partner_addrs = []  # Fill with other node addresses in the cluster

consensus_node = ClusterConsensusNode(self_addr, partner_addrs)
global_decision_engine = GlobalDecisionEngine(consensus_node)
autonomous_executor = AutonomousExecutor(cluster_manager, global_decision_engine)

threading.Thread(target=autonomous_executor.run_loop, daemon=True).start()

console.print("[ONYX] Phase 30: Full Cluster Consensus & Autonomous Decision Making online")

# ==============================
# PHASE 31: Cross-Service AI Optimization & Energy-Aware Orchestration
# ==============================

# ==============================
# 1. Energy-Aware Optimizer
# ==============================
class EnergyOptimizer:
    def __init__(self, cluster_manager, scheduler, reward_system):
        self.cluster_manager = cluster_manager
        self.scheduler = scheduler
        self.reward_system = reward_system
        self.active = True

    def evaluate_node_efficiency(self, node):
        cpu = node.metrics.get("cpu", 0)
        ram = node.metrics.get("ram", 0)
        power = node.metrics.get("power", 50)  # Watts, placeholder
        efficiency = (100 - cpu) + (100 - ram) - power / 10
        return efficiency

    def optimize_services(self):
        nodes = self.cluster_manager.nodes.values()
        for node in nodes:
            efficiency = self.evaluate_node_efficiency(node)
            high_load_services = self.scheduler.get_high_load_services(node.node_id)
            if efficiency < 50 and high_load_services:
                target_node = self.cluster_manager.get_least_loaded_node(exclude_ids=[node.node_id])
                for svc in high_load_services:
                    self.scheduler.migrate_service(svc, source=node.node_id, destination=target_node.node_id)
                    log_manager.info(f"[EnergyOptimizer] Migrated {svc} from {node.node_id} to {target_node.node_id}")
                    self.reward_system.add_points(5, reason="energy_optimized_migration")
            elif efficiency > 80:
                self.scheduler.scale_node_services(node.node_id)
                log_manager.info(f"[EnergyOptimizer] Scaled services on node {node.node_id}")
                self.reward_system.add_points(2, reason="efficiency_scale_up")

# ==============================
# 2. Service Performance Monitor
# ==============================
class CrossServiceMonitor:
    def __init__(self, scheduler, reward_system):
        self.scheduler = scheduler
        self.reward_system = reward_system
        self.active = True

    def monitor_services(self):
        for svc in self.scheduler.list_all_services():
            metrics = self.scheduler.get_service_metrics(svc)
            if metrics["cpu"] > 90:
                log_manager.warn(f"[ServiceMonitor] High CPU on {svc}, triggering optimizer")
                self.reward_system.add_points(1, reason="high_cpu_alert")
            if metrics["ram"] > 90:
                log_manager.warn(f"[ServiceMonitor] High RAM on {svc}, triggering optimizer")
                self.reward_system.add_points(1, reason="high_ram_alert")

# ==============================
# 3. Cross-Service AI Orchestrator
# ==============================
class CrossServiceOrchestrator:
    def __init__(self, cluster_manager, scheduler, reward_system):
        self.energy_optimizer = EnergyOptimizer(cluster_manager, scheduler, reward_system)
        self.service_monitor = CrossServiceMonitor(scheduler, reward_system)
        self.active = True

    def run_loop(self):
        while self.active:
            # Step 1: Monitor services
            self.service_monitor.monitor_services()
            # Step 2: Optimize energy & performance across cluster
            self.energy_optimizer.optimize_services()
            # Step 3: Wait before next optimization cycle
            time.sleep(10)

# ==============================
# 4. Phase 31 Bootstrap
# ==============================
cross_service_orchestrator = CrossServiceOrchestrator(cluster_manager, scheduler, reward_system)
threading.Thread(target=cross_service_orchestrator.run_loop, daemon=True).start()

console.print("[ONYX] Phase 31: Cross-Service AI Optimization & Energy-Aware Orchestration online")

# ==============================
# PHASE 32: Autonomous Failure Recovery & Self-Healing
# ==============================

# ==============================
# 1. Failure Detector
# ==============================
class FailureDetector:
    def __init__(self, cluster_manager, scheduler):
        self.cluster_manager = cluster_manager
        self.scheduler = scheduler
        self.active = True

    def check_node_health(self, node):
        cpu = node.metrics.get("cpu", 0)
        ram = node.metrics.get("ram", 0)
        last_heartbeat = node.metrics.get("last_heartbeat", 0)
        if cpu > 95 or ram > 95 or time.time() - last_heartbeat > 30:
            log_manager.warn(f"[FailureDetector] Node {node.node_id} unhealthy")
            return False
        return True

    def check_service_health(self, service):
        metrics = self.scheduler.get_service_metrics(service)
        if metrics["cpu"] > 95 or metrics["ram"] > 95 or not metrics["alive"]:
            log_manager.warn(f"[FailureDetector] Service {service} failed")
            return False
        return True

# ==============================
# 2. Self-Healing Manager
# ==============================
class SelfHealingManager:
    def __init__(self, cluster_manager, scheduler, infra_healer, backup_manager, reward_system):
        self.cluster_manager = cluster_manager
        self.scheduler = scheduler
        self.infra_healer = infra_healer
        self.backup_manager = backup_manager
        self.reward_system = reward_system
        self.failure_detector = FailureDetector(cluster_manager, scheduler)
        self.active = True

    def recover_service(self, service, node_id):
        try:
            self.scheduler.restart_service(service, node_id)
            log_manager.info(f"[SelfHealing] Restarted service {service} on node {node_id}")
            self.reward_system.add_points(10, reason="service_recovery")
        except Exception as e:
            log_manager.error(f"[SelfHealing] Failed to restart {service} on {node_id}: {e}")
            # Attempt backup restoration
            backup_path = self.backup_manager.get_latest_backup(service)
            if backup_path:
                self.infra_healer.restore_service_from_backup(service, backup_path)
                log_manager.info(f"[SelfHealing] Restored {service} from backup")
                self.reward_system.add_points(15, reason="service_restore_backup")

    def recover_node(self, node):
        try:
            log_manager.info(f"[SelfHealing] Attempting node recovery: {node.node_id}")
            self.infra_healer.reboot_node(node.node_id)
            self.reward_system.add_points(20, reason="node_recovery")
        except Exception as e:
            log_manager.error(f"[SelfHealing] Node {node.node_id} recovery failed: {e}")

    def run_loop(self):
        while self.active:
            # Node-level recovery
            for node in self.cluster_manager.nodes.values():
                if not self.failure_detector.check_node_health(node):
                    self.recover_node(node)
            # Service-level recovery
            for svc in self.scheduler.list_all_services():
                node_id = self.scheduler.get_service_node(svc)
                if not self.failure_detector.check_service_health(svc):
                    self.recover_service(svc, node_id)
            # Wait before next recovery cycle
            time.sleep(15)

# ==============================
# 3. Phase 32 Bootstrap
# ==============================
self_healing_manager = SelfHealingManager(
    cluster_manager=cluster_manager,
    scheduler=scheduler,
    infra_healer=infra_healer,
    backup_manager=backup_manager,
    reward_system=reward_system
)
threading.Thread(target=self_healing_manager.run_loop, daemon=True).start()

console.print("[ONYX] Phase 32: Autonomous Failure Recovery & Self-Healing online")

# ==============================
# PHASE 33: Predictive Maintenance & Proactive Load Balancing
# ==============================

# ==============================
# 1. Predictive Analyzer
# ==============================
class PredictiveAnalyzer:
    def __init__(self, cluster_manager, scheduler):
        self.cluster_manager = cluster_manager
        self.scheduler = scheduler
        self.active = True

    def forecast_node_failure(self, node):
        # Simple linear regression over CPU and RAM trends
        cpu_trend = node.metrics.get("cpu_trend", [])
        ram_trend = node.metrics.get("ram_trend", [])
        if len(cpu_trend) < 5 or len(ram_trend) < 5:
            return False  # Not enough data to predict

        cpu_slope = (cpu_trend[-1] - cpu_trend[0]) / len(cpu_trend)
        ram_slope = (ram_trend[-1] - ram_trend[0]) / len(ram_trend)
        if cpu_slope > 1.5 or ram_slope > 1.5:
            log_manager.warn(f"[PredictiveAnalyzer] Node {node.node_id} likely to fail soon")
            return True
        return False

    def forecast_service_failure(self, service):
        metrics = self.scheduler.get_service_metrics(service)
        cpu = metrics.get("cpu", 0)
        ram = metrics.get("ram", 0)
        latency = metrics.get("latency", 0)
        if cpu > 85 or ram > 85 or latency > 200:
            log_manager.warn(f"[PredictiveAnalyzer] Service {service} may fail soon")
            return True
        return False

# ==============================
# 2. Proactive Load Balancer
# ==============================
class ProactiveLoadBalancer:
    def __init__(self, cluster_manager, scheduler, predictive_analyzer, reward_system):
        self.cluster_manager = cluster_manager
        self.scheduler = scheduler
        self.predictive_analyzer = predictive_analyzer
        self.reward_system = reward_system
        self.active = True

    def redistribute_services(self):
        for svc in self.scheduler.list_all_services():
            node_id = self.scheduler.get_service_node(svc)
            node = self.cluster_manager.nodes[node_id]
            if self.predictive_analyzer.forecast_service_failure(svc):
                target_node = self.find_best_node(exclude=node_id)
                if target_node:
                    self.scheduler.migrate_service(svc, target_node.node_id)
                    log_manager.info(f"[LoadBalancer] Migrated {svc} from {node.node_id} to {target_node.node_id}")
                    self.reward_system.add_points(5, reason="service_migration")

    def find_best_node(self, exclude=None):
        candidates = [n for n in self.cluster_manager.nodes.values() if n.node_id != exclude]
        if not candidates:
            return None
        # Choose node with lowest CPU+RAM combined load
        best_node = min(candidates, key=lambda n: n.metrics.get("cpu", 100) + n.metrics.get("ram", 100))
        return best_node

    def run_loop(self):
        while self.active:
            # Check nodes for potential failures
            for node in self.cluster_manager.nodes.values():
                if self.predictive_analyzer.forecast_node_failure(node):
                    # Preemptively migrate services off this node
                    services = self.scheduler.get_services_on_node(node.node_id)
                    for svc in services:
                        target_node = self.find_best_node(exclude=node.node_id)
                        if target_node:
                            self.scheduler.migrate_service(svc, target_node.node_id)
                            log_manager.info(f"[LoadBalancer] Proactively migrated {svc} from {node.node_id} to {target_node.node_id}")
                            self.reward_system.add_points(10, reason="node_failure_prevention")
            # Service-level proactive balancing
            self.redistribute_services()
            time.sleep(20)

# ==============================
# 3. Phase 33 Bootstrap
# ==============================
predictive_analyzer = PredictiveAnalyzer(cluster_manager=cluster_manager, scheduler=scheduler)
proactive_load_balancer = ProactiveLoadBalancer(
    cluster_manager=cluster_manager,
    scheduler=scheduler,
    predictive_analyzer=predictive_analyzer,
    reward_system=reward_system
)
threading.Thread(target=proactive_load_balancer.run_loop, daemon=True).start()

console.print("[ONYX] Phase 33: Predictive Maintenance & Proactive Load Balancing online")

# ==============================
# PHASE 34: Autonomous Security & Intrusion Prevention
# ==============================

# ==============================
# 1. Threat Detector
# ==============================
class ThreatDetector:
    def __init__(self, network_monitor, traffic_analyzer, ids, reward_system):
        self.network_monitor = network_monitor
        self.traffic_analyzer = traffic_analyzer
        self.ids = ids
        self.reward_system = reward_system
        self.active = True

    def detect_intrusion(self, packet_meta):
        # Simple anomaly detection based on thresholds and past traffic patterns
        src_ip = packet_meta.get("src_ip")
        dst_port = packet_meta.get("dst_port")
        bytes_sent = packet_meta.get("bytes_sent", 0)
        bytes_recv = packet_meta.get("bytes_recv", 0)

        # Example heuristic: sudden spike in traffic
        traffic_pattern = self.traffic_analyzer.get_recent_traffic(src_ip)
        if len(traffic_pattern) >= 5:
            avg_bytes = sum(traffic_pattern[-5:]) / 5
            if bytes_sent > avg_bytes * 3:
                self.ids.flag_suspicious(src_ip, reason="High outbound traffic spike")
                self.reward_system.add_points(5, reason="intrusion_detected")
                return True
        return False

    def run_loop(self):
        while self.active:
            packet_meta = self.network_monitor.capture_packet_meta()
            if packet_meta:
                self.detect_intrusion(packet_meta)
            time.sleep(0.5)

# ==============================
# 2. Intrusion Prevention Engine
# ==============================
class IntrusionPreventionEngine:
    def __init__(self, firewall_manager, vpn_manager, threat_detector, cluster_manager, reward_system):
        self.firewall_manager = firewall_manager
        self.vpn_manager = vpn_manager
        self.threat_detector = threat_detector
        self.cluster_manager = cluster_manager
        self.reward_system = reward_system
        self.active = True

    def neutralize_threat(self, src_ip):
        try:
            # Block via firewall
            self.firewall_manager.block_ip(src_ip)
            log_manager.info(f"[IPS] Blocked malicious IP {src_ip}")
            self.reward_system.add_points(10, reason="threat_neutralized")

            # Optionally re-route traffic via VPN for isolation
            safe_node = self.cluster_manager.get_least_loaded_node()
            if safe_node:
                self.vpn_manager.route_traffic_through_node(src_ip, safe_node.node_id)
                log_manager.info(f"[IPS] Rerouted traffic from {src_ip} through {safe_node.node_id}")
                self.reward_system.add_points(5, reason="traffic_rerouted")
        except Exception as e:
            log_manager.error(f"[IPS] Failed to neutralize {src_ip}: {str(e)}")

    def run_loop(self):
        while self.active:
            for alert in list(self.threat_detector.ids.alerts):
                src_ip = alert.get("src_ip")
                if src_ip:
                    self.neutralize_threat(src_ip)
            time.sleep(1)

# ==============================
# 3. Phase 34 Bootstrap
# ==============================
threat_detector = ThreatDetector(
    network_monitor=network_monitor,
    traffic_analyzer=traffic_analyzer,
    ids=ids,
    reward_system=reward_system
)

ips_engine = IntrusionPreventionEngine(
    firewall_manager=firewall_manager,
    vpn_manager=vpn_manager,
    threat_detector=threat_detector,
    cluster_manager=cluster_manager,
    reward_system=reward_system
)

threading.Thread(target=threat_detector.run_loop, daemon=True).start()
threading.Thread(target=ips_engine.run_loop, daemon=True).start()

console.print("[ONYX] Phase 34: Autonomous Security & Intrusion Prevention online")

# ==============================
# PHASE 35: Self-Healing & Redundancy
# ==============================

# ==============================
# 1. Service Health Monitor
# ==============================
class ServiceHealthMonitor:
    def __init__(self, service_manager, reward_system):
        self.service_manager = service_manager
        self.reward_system = reward_system
        self.active = True

    def check_health(self, service):
        # Check service PID or custom health endpoint
        try:
            if not service.is_running():
                log_manager.warn(f"[Health] Service {service.name} is down")
                return False
            return True
        except Exception as e:
            log_manager.error(f"[Health] Failed to check {service.name}: {str(e)}")
            return False

    def run_loop(self):
        while self.active:
            for svc in self.service_manager.get_all_services():
                if not self.check_health(svc):
                    self.service_manager.restart_service(svc.name)
                    self.reward_system.add_points(5, reason=f"service_{svc.name}_restarted")
            time.sleep(2)

# ==============================
# 2. Redundancy & Failover Engine
# ==============================
class RedundancyEngine:
    def __init__(self, cluster_manager, service_manager, reward_system):
        self.cluster_manager = cluster_manager
        self.service_manager = service_manager
        self.reward_system = reward_system
        self.active = True

    def ensure_redundancy(self):
        # For each critical service, ensure at least N replicas are running
        for svc in self.service_manager.get_all_services():
            if svc.critical:
                replicas_needed = svc.redundancy_level
                current_replicas = self.cluster_manager.count_service_instances(svc.name)
                if current_replicas < replicas_needed:
                    # Launch new instance on least loaded node
                    for _ in range(replicas_needed - current_replicas):
                        target_node = self.cluster_manager.get_least_loaded_node()
                        if target_node:
                            self.cluster_manager.deploy_service_to_node(svc.name, target_node.node_id)
                            log_manager.info(f"[Redundancy] Deployed {svc.name} to {target_node.node_id}")
                            self.reward_system.add_points(10, reason=f"redundancy_{svc.name}_deployed")

    def run_loop(self):
        while self.active:
            self.ensure_redundancy()
            time.sleep(5)

# ==============================
# 3. Phase 35 Bootstrap
# ==============================
service_health_monitor = ServiceHealthMonitor(
    service_manager=service_manager,
    reward_system=reward_system
)

redundancy_engine = RedundancyEngine(
    cluster_manager=cluster_manager,
    service_manager=service_manager,
    reward_system=reward_system
)

threading.Thread(target=service_health_monitor.run_loop, daemon=True).start()
threading.Thread(target=redundancy_engine.run_loop, daemon=True).start()

console.print("[ONYX] Phase 35: Self-Healing & Redundancy for Critical Services online")

# ==============================
# PHASE 36: Predictive Maintenance & Resource Scaling
# ==============================

# ==============================
# 1. Predictive Analyzer
# ==============================
class PredictiveAnalyzer:
    def __init__(self, service_manager):
        self.service_manager = service_manager
        self.active = True
        self.history = {}  # store historical metrics for each service

    def record_metrics(self, service):
        metrics = {
            "cpu": service.get_cpu_usage(),
            "memory": service.get_memory_usage(),
            "latency": service.get_response_time()
        }
        self.history.setdefault(service.name, []).append(metrics)
        if len(self.history[service.name]) > 50:
            self.history[service.name].pop(0)  # keep last 50 metrics

    def predict_failure(self, service_name):
        # Simple linear trend prediction for demonstration
        data = self.history.get(service_name, [])
        if len(data) < 10:
            return False
        avg_cpu = sum(d["cpu"] for d in data) / len(data)
        avg_mem = sum(d["memory"] for d in data) / len(data)
        if avg_cpu > 85 or avg_mem > 85:
            log_manager.warn(f"[Predictive] {service_name} may fail soon (CPU {avg_cpu}%, MEM {avg_mem}%)")
            return True
        return False

    def run_loop(self):
        while self.active:
            for svc in self.service_manager.get_all_services():
                self.record_metrics(svc)
                if self.predict_failure(svc.name):
                    self.service_manager.scale_service(svc.name)
            time.sleep(3)

# ==============================
# 2. Dynamic Resource Scaler
# ==============================
class ResourceScaler:
    def __init__(self, cluster_manager):
        self.cluster_manager = cluster_manager
        self.active = True

    def scale_service_resources(self, service_name, current_node):
        # Increase CPU/RAM limits on predicted high-load nodes
        node_load = current_node.get_load()
        if node_load["cpu"] > 80:
            current_node.adjust_service(service_name, cpu_quota=1.5)
            log_manager.info(f"[Scaler] Increased CPU quota for {service_name} on {current_node.node_id}")
        if node_load["memory"] > 80:
            current_node.adjust_service(service_name, memory_quota=1.5)
            log_manager.info(f"[Scaler] Increased memory quota for {service_name} on {current_node.node_id}")

    def run_loop(self):
        while self.active:
            for node in self.cluster_manager.get_all_nodes():
                for svc in node.get_running_services():
                    self.scale_service_resources(svc.name, node)
            time.sleep(5)

# ==============================
# 3. Phase 36 Bootstrap
# ==============================
predictive_analyzer = PredictiveAnalyzer(service_manager=service_manager)
resource_scaler = ResourceScaler(cluster_manager=cluster_manager)

threading.Thread(target=predictive_analyzer.run_loop, daemon=True).start()
threading.Thread(target=resource_scaler.run_loop, daemon=True).start()

console.print("[ONYX] Phase 36: Predictive Maintenance & Resource Scaling online")

# ==============================
# PHASE 37: Autonomous Security Adaptation
# ==============================

# ==============================
# 1. Adaptive Security Engine
# ==============================
class SecurityAdapter:
    def __init__(self, firewall_manager, vpn_manager, ids_system):
        self.firewall = firewall_manager
        self.vpn = vpn_manager
        self.ids = ids_system
        self.active = True
        self.threat_history = {}  # track detected threats per host

    def analyze_threats(self):
        alerts = self.ids.get_recent_alerts()
        for alert in alerts:
            host = alert.get("source_ip")
            self.threat_history.setdefault(host, []).append(alert)
            if len(self.threat_history[host]) > 20:
                self.threat_history[host].pop(0)

            self.adapt_defenses(host)

    def adapt_defenses(self, host):
        threat_count = len(self.threat_history.get(host, []))
        if threat_count > 5:
            # Apply dynamic firewall rule
            self.firewall.block_ip(host)
            log_manager.warn(f"[SecurityAdapter] Host {host} blocked due to {threat_count} alerts")

            # Apply VPN isolation if critical
            if threat_count > 10:
                self.vpn.disconnect(host)
                log_manager.warn(f"[SecurityAdapter] Host {host} disconnected from VPN")

    def run_loop(self):
        while self.active:
            self.analyze_threats()
            time.sleep(2)

# ==============================
# 2. IDS Mock (for demonstration)
# ==============================
class IDS:
    def __init__(self):
        self.alerts = []

    def generate_alert(self, source_ip, severity):
        alert = {"source_ip": source_ip, "severity": severity, "timestamp": time.time()}
        self.alerts.append(alert)

    def get_recent_alerts(self):
        recent = [a for a in self.alerts if time.time() - a["timestamp"] < 60]
        self.alerts = recent
        return recent

# ==============================
# 3. Firewall & VPN Stub Implementations
# ==============================
class FirewallManager:
    def block_ip(self, ip):
        # Normally this would call nftables or ufw; here we log
        log_manager.info(f"[Firewall] Blocking IP {ip}")

class VPNManager:
    def disconnect(self, host):
        # Normally disconnect from WireGuard; here we log
        log_manager.info(f"[VPN] Disconnecting host {host}")

# ==============================
# 4. Phase 37 Bootstrap
# ==============================
ids_system = IDS()
firewall_manager = FirewallManager()
vpn_manager = VPNManager()
security_adapter = SecurityAdapter(firewall_manager, vpn_manager, ids_system)

threading.Thread(target=security_adapter.run_loop, daemon=True).start()

# Demonstration: simulate threat alerts
threading.Thread(
    target=lambda: [ids_system.generate_alert(f"192.168.1.{i%10}", severity="high") or time.sleep(1) for i in range(50)],
    daemon=True
).start()

console.print("[ONYX] Phase 37: Autonomous Security Adaptation online")

# ==============================
# PHASE 38: Autonomous Patch & Update Management
# ==============================

# ==============================
# 1. Patch Manager Core
# ==============================
class PatchManager:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.active = True
        self.update_queue = []
        self.lock = threading.Lock()
        self.last_run = 0

    def check_system_updates(self):
        # Check OS packages (Debian/Ubuntu example)
        try:
            result = subprocess.run(["apt", "list", "--upgradable"], capture_output=True, text=True)
            updates = [line.split("/")[0] for line in result.stdout.splitlines()[1:] if line]
            self.logger.info(f"[PatchManager] OS updates found: {updates}")
            return updates
        except Exception as e:
            self.logger.error(f"[PatchManager] Failed to check OS updates: {e}")
            return []

    def check_python_updates(self):
        # Use pip to check outdated packages
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                                    capture_output=True, text=True)
            import json
            updates = [pkg["name"] for pkg in json.loads(result.stdout)]
            self.logger.info(f"[PatchManager] Python updates found: {updates}")
            return updates
        except Exception as e:
            self.logger.error(f"[PatchManager] Failed to check Python updates: {e}")
            return []

    def queue_update(self, package_name, update_type="os"):
        with self.lock:
            self.update_queue.append({"package": package_name, "type": update_type})
            self.logger.info(f"[PatchManager] Queued {update_type} update for {package_name}")

    def apply_update(self, update_entry):
        package = update_entry["package"]
        update_type = update_entry["type"]
        try:
            if update_type == "os":
                subprocess.run(["sudo", "apt", "install", "-y", package], check=True)
            elif update_type == "python":
                subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package], check=True)
            self.logger.info(f"[PatchManager] Applied {update_type} update: {package}")
        except Exception as e:
            self.logger.error(f"[PatchManager] Failed to apply {update_type} update {package}: {e}")

    def run_loop(self):
        while self.active:
            # Run update check every configured interval
            if time.time() - self.last_run > self.config.get("patch_interval", 3600):
                self.last_run = time.time()
                os_updates = self.check_system_updates()
                py_updates = self.check_python_updates()
                for pkg in os_updates:
                    self.queue_update(pkg, "os")
                for pkg in py_updates:
                    self.queue_update(pkg, "python")
            # Apply queued updates
            with self.lock:
                while self.update_queue:
                    update_entry = self.update_queue.pop(0)
                    self.apply_update(update_entry)
            time.sleep(10)

# ==============================
# 2. Phase 38 Bootstrap
# ==============================
patch_manager = PatchManager(config=config, logger=log_manager)
threading.Thread(target=patch_manager.run_loop, daemon=True).start()

console.print("[ONYX] Phase 38: Autonomous Patch & Update Management online")

# ==============================
# PHASE 39: Autonomous Resource Balancing
# ==============================

# ==============================
# 1. Resource Monitor
# ==============================
class ResourceMonitor:
    def __init__(self, logger):
        self.logger = logger
        self.metrics = {
            "cpu": psutil.cpu_percent(interval=1),
            "ram": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage("/").percent,
            "network_sent": 0,
            "network_recv": 0,
            "gpu": 0  # Placeholder for GPU % usage; integrate NVML later
        }
        self.active = True
        self.prev_net = psutil.net_io_counters()

    def update_metrics(self):
        self.metrics["cpu"] = psutil.cpu_percent(interval=None)
        self.metrics["ram"] = psutil.virtual_memory().percent
        self.metrics["disk"] = psutil.disk_usage("/").percent
        net = psutil.net_io_counters()
        self.metrics["network_sent"] = net.bytes_sent - self.prev_net.bytes_sent
        self.metrics["network_recv"] = net.bytes_recv - self.prev_net.bytes_recv
        self.prev_net = net
        # TODO: Add GPU integration via pynvml if available

# ==============================
# 2. Resource Balancer
# ==============================
class ResourceBalancer:
    def __init__(self, logger, monitor, agents, services, vms):
        self.logger = logger
        self.monitor = monitor
        self.agents = agents
        self.services = services
        self.vms = vms
        self.active = True

    def redistribute_load(self):
        # Example logic: throttle non-critical services if CPU > 85%
        cpu_usage = self.monitor.metrics["cpu"]
        ram_usage = self.monitor.metrics["ram"]
        for svc in self.services:
            if cpu_usage > 85 or ram_usage > 90:
                if not getattr(svc, "critical", False):
                    self.logger.info(f"[ResourceBalancer] Throttling service {svc.name} due to high load")
                    svc.throttle()
            else:
                svc.resume()

        # Agents: reduce idle agent processing if RAM is high
        for agent in self.agents:
            if ram_usage > 90:
                agent.pause_learning()
            else:
                agent.resume_learning()

        # VM balancing: migrate or pause VMs if host overloaded
        for vm in self.vms:
            if cpu_usage > 90 or ram_usage > 95:
                vm.pause()
            else:
                vm.resume()

    def run_loop(self):
        while self.active:
            self.monitor.update_metrics()
            self.redistribute_load()
            self.logger.info(f"[ResourceBalancer] Metrics: {self.monitor.metrics}")
            time.sleep(5)  # adjust interval as needed

# ==============================
# 3. Phase 39 Bootstrap
# ==============================
resource_monitor = ResourceMonitor(logger=log_manager)
resource_balancer = ResourceBalancer(
    logger=log_manager,
    monitor=resource_monitor,
    agents=ai_agents,         # previously defined AIAgent instances
    services=service_manager.services,  # previously registered services
    vms=vm_manager.vms        # VM instances
)

threading.Thread(target=resource_balancer.run_loop, daemon=True).start()

console.print("[ONYX] Phase 39: Autonomous Resource Balancing online")

# ==============================
# PHASE 40: Predictive Workload Scheduling
# ==============================

import numpy as np
from sklearn.linear_model import LinearRegression
from collections import deque

# ==============================
# 1. Historical Metrics Collector
# ==============================
class MetricsHistory:
    def __init__(self, max_len=3600):  # store last hour of metrics at 1s intervals
        self.cpu_history = deque(maxlen=max_len)
        self.ram_history = deque(maxlen=max_len)
        self.disk_history = deque(maxlen=max_len)
        self.network_sent_history = deque(maxlen=max_len)
        self.network_recv_history = deque(maxlen=max_len)
        self.timestamps = deque(maxlen=max_len)

    def add_metrics(self, metrics):
        now = time.time()
        self.cpu_history.append(metrics["cpu"])
        self.ram_history.append(metrics["ram"])
        self.disk_history.append(metrics["disk"])
        self.network_sent_history.append(metrics["network_sent"])
        self.network_recv_history.append(metrics["network_recv"])
        self.timestamps.append(now)

    def get_training_data(self):
        # Convert history to numpy arrays for prediction
        X = np.array(list(self.timestamps)).reshape(-1, 1)
        y_cpu = np.array(list(self.cpu_history))
        y_ram = np.array(list(self.ram_history))
        return X, y_cpu, y_ram

# ==============================
# 2. Predictive Scheduler
# ==============================
class PredictiveScheduler:
    def __init__(self, logger, monitor, history, resource_balancer):
        self.logger = logger
        self.monitor = monitor
        self.history = history
        self.resource_balancer = resource_balancer
        self.active = True
        self.model_cpu = LinearRegression()
        self.model_ram = LinearRegression()

    def train_models(self):
        X, y_cpu, y_ram = self.history.get_training_data()
        if len(X) > 10:  # minimal data to train
            self.model_cpu.fit(X, y_cpu)
            self.model_ram.fit(X, y_ram)

    def predict_next_load(self, seconds_ahead=60):
        future_time = np.array([[time.time() + seconds_ahead]])
        pred_cpu = self.model_cpu.predict(future_time)[0]
        pred_ram = self.model_ram.predict(future_time)[0]
        return {"cpu": pred_cpu, "ram": pred_ram}

    def schedule_tasks(self):
        prediction = self.predict_next_load()
        cpu_pred = prediction["cpu"]
        ram_pred = prediction["ram"]

        # Preemptively throttle or defer low-priority services
        for svc in self.resource_balancer.services:
            if cpu_pred > 85 or ram_pred > 90:
                if not getattr(svc, "critical", False):
                    self.logger.info(f"[PredictiveScheduler] Deferring {svc.name} due to predicted high load")
                    svc.throttle()
            else:
                svc.resume()

        # Preemptively pause low-priority agents if predicted RAM spike
        for agent in self.resource_balancer.agents:
            if ram_pred > 90:
                agent.pause_learning()
            else:
                agent.resume_learning()

        # Future: Can also pre-scale cluster nodes or schedule VM tasks
        self.logger.info(f"[PredictiveScheduler] Predicted CPU: {cpu_pred:.1f}%, RAM: {ram_pred:.1f}%")

    def run_loop(self):
        while self.active:
            self.history.add_metrics(self.monitor.metrics)
            self.train_models()
            self.schedule_tasks()
            time.sleep(5)

# ==============================
# 3. Phase 40 Bootstrap
# ==============================
metrics_history = MetricsHistory()
predictive_scheduler = PredictiveScheduler(
    logger=log_manager,
    monitor=resource_monitor,
    history=metrics_history,
    resource_balancer=resource_balancer
)

threading.Thread(target=predictive_scheduler.run_loop, daemon=True).start()

console.print("[ONYX] Phase 40: Predictive Workload Scheduling online")

# ==============================
# PHASE 41: Adaptive Cluster Scaling
# ==============================

import uuid
from collections import defaultdict
import subprocess

# ==============================
# 1. Cluster Node Representation
# ==============================
class ClusterNode:
    def __init__(self, node_id=None, role="worker"):
        self.node_id = node_id or str(uuid.uuid4())
        self.role = role
        self.status = "idle"
        self.metrics = {"cpu": 0.0, "ram": 0.0}
        self.last_heartbeat = time.time()

    def update_metrics(self, cpu, ram):
        self.metrics["cpu"] = cpu
        self.metrics["ram"] = ram
        self.last_heartbeat = time.time()

    def is_active(self):
        return (time.time() - self.last_heartbeat) < 30  # active if heartbeat within 30s

# ==============================
# 2. Cluster Manager
# ==============================
class AdaptiveClusterManager:
    def __init__(self, logger, predictive_scheduler, max_nodes=10, min_nodes=1):
        self.logger = logger
        self.predictive_scheduler = predictive_scheduler
        self.nodes = {}
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.active = True

    def add_node(self):
        if len(self.nodes) >= self.max_nodes:
            self.logger.info("[ClusterManager] Max nodes reached, cannot add more.")
            return None
        node = ClusterNode(role="worker")
        self.nodes[node.node_id] = node
        self.logger.info(f"[ClusterManager] Added new node: {node.node_id}")
        return node

    def remove_node(self):
        # Remove idle node with lowest utilization
        removable = [n for n in self.nodes.values() if n.is_active() and n.status == "idle"]
        if removable:
            node = min(removable, key=lambda n: n.metrics["cpu"] + n.metrics["ram"])
            del self.nodes[node.node_id]
            self.logger.info(f"[ClusterManager] Removed node: {node.node_id}")
            return node
        self.logger.info("[ClusterManager] No idle node available to remove")
        return None

    def monitor_cluster(self):
        while self.active:
            prediction = self.predictive_scheduler.predict_next_load()
            cpu_pred = prediction["cpu"]
            ram_pred = prediction["ram"]

            # Simple scaling logic
            if cpu_pred > 80 or ram_pred > 85:
                self.add_node()
            elif cpu_pred < 40 and ram_pred < 50:
                if len(self.nodes) > self.min_nodes:
                    self.remove_node()

            # Heartbeat update simulation
            for node in self.nodes.values():
                node.update_metrics(cpu=np.random.uniform(10, 70), ram=np.random.uniform(20, 60))
            time.sleep(10)

# ==============================
# 3. Phase 41 Bootstrap
# ==============================
cluster_manager = AdaptiveClusterManager(
    logger=log_manager,
    predictive_scheduler=predictive_scheduler,
    max_nodes=10,
    min_nodes=1
)

threading.Thread(target=cluster_manager.monitor_cluster, daemon=True).start()

console.print("[ONYX] Phase 41: Adaptive Cluster Scaling online")

# ==============================
# PHASE 42: Automated Load Balancing & Task Distribution
# ==============================

import queue

# ==============================
# 1. Task Representation
# ==============================
class Task:
    def __init__(self, task_id=None, payload=None, cpu_req=10, ram_req=20):
        self.task_id = task_id or str(uuid.uuid4())
        self.payload = payload
        self.cpu_req = cpu_req
        self.ram_req = ram_req
        self.assigned_node = None
        self.status = "pending"  # pending, running, completed

# ==============================
# 2. Load Balancer
# ==============================
class LoadBalancer:
    def __init__(self, cluster_manager, logger):
        self.cluster_manager = cluster_manager
        self.logger = logger
        self.task_queue = queue.Queue()
        self.active = True

    def assign_task(self, task):
        # Choose best node based on current load + predicted load
        active_nodes = [n for n in self.cluster_manager.nodes.values() if n.is_active()]
        if not active_nodes:
            self.logger.warning(f"[LoadBalancer] No active nodes for task {task.task_id}")
            return False
        # Node score: CPU + RAM usage, lower is better
        def node_score(node):
            return node.metrics["cpu"] + node.metrics["ram"]
        best_node = min(active_nodes, key=node_score)
        task.assigned_node = best_node.node_id
        task.status = "running"
        best_node.status = "busy"
        self.logger.info(f"[LoadBalancer] Assigned task {task.task_id} to node {best_node.node_id}")
        return True

    def distribute_tasks(self):
        while self.active:
            try:
                task = self.task_queue.get(timeout=5)
                if not self.assign_task(task):
                    # Requeue task if no node available
                    self.task_queue.put(task)
            except queue.Empty:
                time.sleep(1)

    def submit_task(self, payload, cpu_req=10, ram_req=20):
        task = Task(payload=payload, cpu_req=cpu_req, ram_req=ram_req)
        self.task_queue.put(task)
        self.logger.info(f"[LoadBalancer] Submitted task {task.task_id}")
        return task

# ==============================
# 3. Phase 42 Bootstrap
# ==============================
load_balancer = LoadBalancer(cluster_manager, log_manager)
threading.Thread(target=load_balancer.distribute_tasks, daemon=True).start()

console.print("[ONYX] Phase 42: Automated Load Balancing online")

# ==============================
# 4. Example Usage
# ==============================
for i in range(5):
    load_balancer.submit_task(payload=f"Compute job {i}", cpu_req=15, ram_req=25)

# ==============================
# PHASE 43: Fault-Tolerant Task Execution & Node Recovery
# ==============================

# ==============================
# 1. Node Wrapper with Health & Recovery
# ==============================
class ClusterNode:
    def __init__(self, node_id, metrics=None):
        self.node_id = node_id
        self.metrics = metrics or {"cpu": 0, "ram": 0}
        self.status = "active"  # active, busy, offline
        self.tasks_running = {}
        self.last_heartbeat = time.time()

    def is_active(self):
        return self.status == "active"

    def heartbeat(self, metrics):
        self.metrics.update(metrics)
        self.last_heartbeat = time.time()
        if self.status == "offline":
            self.status = "active"
            console.print(f"[ClusterNode] Node {self.node_id} recovered via heartbeat")

    def mark_offline(self):
        self.status = "offline"
        console.print(f"[ClusterNode] Node {self.node_id} marked offline")

    def assign_task(self, task):
        self.tasks_running[task.task_id] = task
        task.assigned_node = self.node_id
        task.status = "running"
        self.status = "busy"

    def complete_task(self, task_id):
        if task_id in self.tasks_running:
            self.tasks_running[task_id].status = "completed"
            del self.tasks_running[task_id]
            self.status = "active"

# ==============================
# 2. Task Recovery Engine
# ==============================
class FaultTolerantScheduler:
    def __init__(self, cluster_manager, load_balancer, logger):
        self.cluster_manager = cluster_manager
        self.load_balancer = load_balancer
        self.logger = logger
        self.active = True

    def monitor_nodes(self):
        while self.active:
            current_time = time.time()
            for node in self.cluster_manager.nodes.values():
                # If no heartbeat for >15s, mark offline
                if current_time - node.last_heartbeat > 15:
                    if node.is_active():
                        node.mark_offline()
                        self.logger.warning(f"[FaultTolerantScheduler] Node {node.node_id} unresponsive. Rescheduling tasks...")
                        self.reschedule_tasks(node)
            time.sleep(5)

    def reschedule_tasks(self, node):
        for task_id, task in node.tasks_running.items():
            task.status = "pending"
            task.assigned_node = None
            self.load_balancer.task_queue.put(task)
            self.logger.info(f"[FaultTolerantScheduler] Task {task_id} rescheduled from node {node.node_id}")
        node.tasks_running.clear()

# ==============================
# 3. Phase 43 Bootstrap
# ==============================
fault_scheduler = FaultTolerantScheduler(cluster_manager, load_balancer, log_manager)
threading.Thread(target=fault_scheduler.monitor_nodes, daemon=True).start()

console.print("[ONYX] Phase 43: Fault-Tolerant Scheduler online")

# ==============================
# 4. Simulate Node Failure
# ==============================
# Example: Mark a node offline after 10s to test rescheduling
def simulate_node_failure(node_id, delay=10):
    time.sleep(delay)
    if node_id in cluster_manager.nodes:
        cluster_manager.nodes[node_id].mark_offline()

threading.Thread(target=simulate_node_failure, args=(list(cluster_manager.nodes.keys())[0],), daemon=True).start()

# ==============================
# PHASE 44: Predictive Load Management & Preemptive Task Migration
# ==============================

import numpy as np
from collections import deque

# ==============================
# 1. Node Metrics History
# ==============================
class NodeMetricsHistory:
    def __init__(self, maxlen=50):
        self.cpu_history = deque(maxlen=maxlen)
        self.ram_history = deque(maxlen=maxlen)

    def add_metrics(self, cpu, ram):
        self.cpu_history.append(cpu)
        self.ram_history.append(ram)

    def predict_next_load(self):
        # Simple linear regression for next CPU/RAM
        if len(self.cpu_history) < 2:
            return {"cpu": self.cpu_history[-1] if self.cpu_history else 0,
                    "ram": self.ram_history[-1] if self.ram_history else 0}
        x = np.arange(len(self.cpu_history))
        cpu_fit = np.polyfit(x, list(self.cpu_history), 1)
        ram_fit = np.polyfit(x, list(self.ram_history), 1)
        return {
            "cpu": float(cpu_fit[0]*len(x) + cpu_fit[1]),
            "ram": float(ram_fit[0]*len(x) + ram_fit[1])
        }

# Attach history to each cluster node
for node in cluster_manager.nodes.values():
    node.metrics_history = NodeMetricsHistory()

# ==============================
# 2. Preemptive Migration Engine
# ==============================
class PredictiveLoadManager:
    def __init__(self, cluster_manager, load_balancer, logger, cpu_threshold=85, ram_threshold=85):
        self.cluster_manager = cluster_manager
        self.load_balancer = load_balancer
        self.logger = logger
        self.cpu_threshold = cpu_threshold
        self.ram_threshold = ram_threshold
        self.active = True

    def monitor_and_migrate(self):
        while self.active:
            for node in self.cluster_manager.nodes.values():
                if not node.is_active() or not hasattr(node, "metrics_history"):
                    continue

                # Update metrics history
                node.metrics_history.add_metrics(node.metrics.get("cpu",0), node.metrics.get("ram",0))

                # Predict next load
                predicted = node.metrics_history.predict_next_load()
                if predicted["cpu"] > self.cpu_threshold or predicted["ram"] > self.ram_threshold:
                    self.logger.info(f"[PredictiveLoadManager] Node {node.node_id} predicted to overload (CPU: {predicted['cpu']:.1f}%, RAM: {predicted['ram']:.1f}%). Preemptively migrating tasks...")
                    self.preemptive_migrate(node)
            time.sleep(5)

    def preemptive_migrate(self, node):
        # Move half of running tasks to other nodes
        tasks_to_migrate = list(node.tasks_running.values())[:len(node.tasks_running)//2]
        for task in tasks_to_migrate:
            task.status = "pending"
            task.assigned_node = None
            self.load_balancer.task_queue.put(task)
            del node.tasks_running[task.task_id]
            self.logger.info(f"[PredictiveLoadManager] Task {task.task_id} migrated from Node {node.node_id}")

# ==============================
# 3. Phase 44 Bootstrap
# ==============================
predictive_manager = PredictiveLoadManager(cluster_manager, load_balancer, log_manager)
threading.Thread(target=predictive_manager.monitor_and_migrate, daemon=True).start()
console.print("[ONYX] Phase 44: Predictive Load Manager online")

# ==============================
# PHASE 45: Autonomous Multi-Agent Task Optimization
# ==============================

import heapq
from enum import Enum

# ==============================
# 1. Task Priority Levels
# ==============================
class TaskPriority(Enum):
    CRITICAL = 3
    HIGH = 2
    NORMAL = 1
    LOW = 0

# ==============================
# 2. Task Wrapper with Priority
# ==============================
class OptimizableTask:
    def __init__(self, task_id, func, cpu_req=1, ram_req=1, priority=TaskPriority.NORMAL):
        self.task_id = task_id
        self.func = func
        self.cpu_req = cpu_req
        self.ram_req = ram_req
        self.priority = priority
        self.status = "pending"
        self.assigned_node = None

    def __lt__(self, other):
        # For priority queue: higher priority runs first
        return self.priority.value > other.priority.value

# ==============================
# 3. Multi-Agent Optimizer
# ==============================
class MultiAgentOptimizer:
    def __init__(self, cluster_manager, logger):
        self.cluster_manager = cluster_manager
        self.logger = logger
        self.task_queue = []
        self.active = True

    def submit_task(self, task):
        heapq.heappush(self.task_queue, task)
        self.logger.info(f"[MultiAgentOptimizer] Task {task.task_id} submitted with priority {task.priority.name}")

    def optimize_allocation(self):
        while self.active:
            if not self.task_queue:
                time.sleep(1)
                continue

            task = heapq.heappop(self.task_queue)
            node = self.select_best_node(task)
            if node:
                task.assigned_node = node.node_id
                task.status = "running"
                node.tasks_running[task.task_id] = task
                threading.Thread(target=self.run_task_on_node, args=(node, task)).start()
                self.logger.info(f"[MultiAgentOptimizer] Task {task.task_id} assigned to Node {node.node_id}")
            else:
                # No suitable node, requeue with backoff
                task.status = "pending"
                heapq.heappush(self.task_queue, task)
                self.logger.warning(f"[MultiAgentOptimizer] No suitable node for Task {task.task_id}, requeued")
                time.sleep(2)

    def select_best_node(self, task):
        best_node = None
        min_load = float("inf")
        for node in self.cluster_manager.nodes.values():
            if not node.is_active():
                continue
            cpu_available = 100 - node.metrics.get("cpu", 0)
            ram_available = 100 - node.metrics.get("ram", 0)
            if cpu_available >= task.cpu_req and ram_available >= task.ram_req:
                node_load = node.metrics.get("cpu",0) + node.metrics.get("ram",0)
                if node_load < min_load:
                    min_load = node_load
                    best_node = node
        return best_node

    def run_task_on_node(self, node, task):
        try:
            task.func()
            task.status = "completed"
            self.logger.info(f"[MultiAgentOptimizer] Task {task.task_id} completed on Node {node.node_id}")
        except Exception as e:
            task.status = "failed"
            self.logger.error(f"[MultiAgentOptimizer] Task {task.task_id} failed on Node {node.node_id}: {str(e)}")
        finally:
            if task.task_id in node.tasks_running:
                del node.tasks_running[task.task_id]

# ==============================
# 4. Phase 45 Bootstrap
# ==============================
multi_agent_optimizer = MultiAgentOptimizer(cluster_manager, log_manager)
threading.Thread(target=multi_agent_optimizer.optimize_allocation, daemon=True).start()
console.print("[ONYX] Phase 45: Multi-Agent Task Optimizer online")

# ==============================
# 5. Example Task Submissions
# ==============================
def example_backup_task():
    log_manager.info("[Task] Performing backup task...")
    time.sleep(3)  # Simulate work

def example_analysis_task():
    log_manager.info("[Task] Performing analytics task...")
    time.sleep(2)

multi_agent_optimizer.submit_task(OptimizableTask("backup_001", example_backup_task, cpu_req=10, ram_req=20, priority=TaskPriority.CRITICAL))
multi_agent_optimizer.submit_task(OptimizableTask("analysis_001", example_analysis_task, cpu_req=5, ram_req=10, priority=TaskPriority.HIGH))

# ==============================
# PHASE 46: Reinforcement-Based Resource Reallocation
# ==============================

import random

# ==============================
# 1. RL Agent for Resource Management
# ==============================
class RLResourceAgent:
    def __init__(self, cluster_manager, logger, learning_rate=0.1, discount_factor=0.9, exploration=0.2):
        self.cluster_manager = cluster_manager
        self.logger = logger
        self.q_table = {}  # {(node_id, task_type): q_value}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration = exploration
        self.active = True

    def choose_node(self, task_type):
        # Epsilon-greedy selection
        if random.random() < self.exploration:
            node = random.choice(list(self.cluster_manager.nodes.values()))
            self.logger.debug(f"[RLAgent] Randomly selected Node {node.node_id} for task {task_type}")
            return node
        # Greedy selection based on Q-values
        best_q = -float('inf')
        best_node = None
        for node_id, node in self.cluster_manager.nodes.items():
            q_value = self.q_table.get((node_id, task_type), 0)
            if q_value > best_q and node.is_active():
                best_q = q_value
                best_node = node
        if best_node:
            self.logger.debug(f"[RLAgent] Selected Node {best_node.node_id} for task {task_type} with Q={best_q}")
        return best_node or random.choice(list(self.cluster_manager.nodes.values()))

    def update_q_value(self, node_id, task_type, reward):
        old_q = self.q_table.get((node_id, task_type), 0)
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * old_q - old_q)
        self.q_table[(node_id, task_type)] = new_q
        self.logger.debug(f"[RLAgent] Updated Q-value for Node {node_id}, task {task_type}: {new_q:.2f}")

# ==============================
# 2. Reward System Integration
# ==============================
class ClusterRewardSystem:
    def __init__(self, logger):
        self.logger = logger

    def evaluate_task(self, node, task, duration):
        # Reward inversely proportional to time taken and load
        cpu_load = node.metrics.get("cpu", 50)
        ram_load = node.metrics.get("ram", 50)
        reward = max(0, 100 - duration - cpu_load - ram_load)
        self.logger.debug(f"[RewardSystem] Task {task.task_id} on Node {node.node_id} earned reward {reward}")
        return reward

# ==============================
# 3. RL Scheduler Integration
# ==============================
class RLScheduler:
    def __init__(self, optimizer, rl_agent, reward_system, logger):
        self.optimizer = optimizer
        self.rl_agent = rl_agent
        self.reward_system = reward_system
        self.logger = logger
        self.active = True

    def run(self):
        while self.active:
            if not self.optimizer.task_queue:
                time.sleep(1)
                continue
            task = heapq.heappop(self.optimizer.task_queue)
            node = self.rl_agent.choose_node(task.func.__name__)
            if node:
                start_time = time.time()
                task.status = "running"
                node.tasks_running[task.task_id] = task
                try:
                    task.func()
                    duration = time.time() - start_time
                    reward = self.reward_system.evaluate_task(node, task, duration)
                    self.rl_agent.update_q_value(node.node_id, task.func.__name__, reward)
                    task.status = "completed"
                    self.logger.info(f"[RLScheduler] Task {task.task_id} completed on Node {node.node_id} in {duration:.2f}s")
                except Exception as e:
                    task.status = "failed"
                    self.logger.error(f"[RLScheduler] Task {task.task_id} failed on Node {node.node_id}: {e}")
                finally:
                    if task.task_id in node.tasks_running:
                        del node.tasks_running[task.task_id]
            else:
                # Requeue task if no node available
                heapq.heappush(self.optimizer.task_queue, task)
                time.sleep(2)

# ==============================
# 4. Phase 46 Bootstrap
# ==============================
reward_system = ClusterRewardSystem(log_manager)
rl_agent = RLResourceAgent(cluster_manager, log_manager)
rl_scheduler = RLScheduler(multi_agent_optimizer, rl_agent, reward_system, log_manager)

threading.Thread(target=rl_scheduler.run, daemon=True).start()
console.print("[ONYX] Phase 46: Reinforcement-Based Resource Scheduler online")

# ==============================
# 5. Example Task Submission
# ==============================
def heavy_analysis_task():
    log_manager.info("[Task] Performing heavy analytics...")
    time.sleep(5)

def quick_backup_task():
    log_manager.info("[Task] Performing quick backup...")
    time.sleep(2)

multi_agent_optimizer.submit_task(OptimizableTask("analysis_002", heavy_analysis_task, cpu_req=20, ram_req=30, priority=TaskPriority.HIGH))
multi_agent_optimizer.submit_task(OptimizableTask("backup_002", quick_backup_task, cpu_req=5, ram_req=10, priority=TaskPriority.CRITICAL))

# ==============================
# PHASE 47: Predictive Load Balancing
# ==============================

import numpy as np
from collections import deque

# ==============================
# 1. Node Load Predictor
# ==============================
class NodeLoadPredictor:
    def __init__(self, window_size=10, logger=None):
        self.window_size = window_size
        self.cpu_history = {}  # node_id -> deque
        self.ram_history = {}  # node_id -> deque
        self.logger = logger or log_manager

    def record_metrics(self, node_id, cpu, ram):
        if node_id not in self.cpu_history:
            self.cpu_history[node_id] = deque(maxlen=self.window_size)
            self.ram_history[node_id] = deque(maxlen=self.window_size)
        self.cpu_history[node_id].append(cpu)
        self.ram_history[node_id].append(ram)
        self.logger.debug(f"[Predictor] Recorded metrics Node {node_id}: CPU={cpu} RAM={ram}")

    def predict_next(self, node_id):
        cpu_next = np.mean(self.cpu_history.get(node_id, [50]))
        ram_next = np.mean(self.ram_history.get(node_id, [50]))
        self.logger.debug(f"[Predictor] Predicted next load Node {node_id}: CPU={cpu_next:.2f}, RAM={ram_next:.2f}")
        return cpu_next, ram_next

# ==============================
# 2. Predictive Load Balancer
# ==============================
class PredictiveLoadBalancer:
    def __init__(self, cluster_manager, rl_agent, predictor, logger, cpu_threshold=85, ram_threshold=85):
        self.cluster_manager = cluster_manager
        self.rl_agent = rl_agent
        self.predictor = predictor
        self.cpu_threshold = cpu_threshold
        self.ram_threshold = ram_threshold
        self.logger = logger
        self.active = True

    def run(self):
        while self.active:
            for node_id, node in self.cluster_manager.nodes.items():
                if not node.is_active():
                    continue
                predicted_cpu, predicted_ram = self.predictor.predict_next(node_id)
                if predicted_cpu > self.cpu_threshold or predicted_ram > self.ram_threshold:
                    self.logger.warning(f"[LoadBalancer] Node {node_id} predicted to exceed threshold: CPU={predicted_cpu:.1f}, RAM={predicted_ram:.1f}")
                    # Migrate low-priority tasks
                    tasks_to_move = [t for t in node.tasks_running.values() if t.priority.value < TaskPriority.HIGH.value]
                    for task in tasks_to_move:
                        self.migrate_task(task, node)
            time.sleep(3)

    def migrate_task(self, task, from_node):
        self.logger.info(f"[LoadBalancer] Migrating Task {task.task_id} from Node {from_node.node_id}")
        new_node = self.rl_agent.choose_node(task.func.__name__)
        if new_node and new_node.node_id != from_node.node_id:
            from_node.tasks_running.pop(task.task_id, None)
            new_node.tasks_running[task.task_id] = task
            self.logger.info(f"[LoadBalancer] Task {task.task_id} moved to Node {new_node.node_id}")
        else:
            self.logger.debug(f"[LoadBalancer] No suitable node found to migrate Task {task.task_id}")

# ==============================
# 3. Phase 47 Bootstrap
# ==============================
predictor = NodeLoadPredictor(window_size=10, logger=log_manager)
predictive_balancer = PredictiveLoadBalancer(cluster_manager, rl_agent, predictor, log_manager)

threading.Thread(target=predictive_balancer.run, daemon=True).start()
console.print("[ONYX] Phase 47: Predictive Load Balancer online")

# ==============================
# 4. Example Metrics Update Loop
# ==============================
def simulate_node_metrics():
    while True:
        for node_id, node in cluster_manager.nodes.items():
            cpu = random.randint(20, 90)
            ram = random.randint(20, 90)
            node.metrics['cpu'] = cpu
            node.metrics['ram'] = ram
            predictor.record_metrics(node_id, cpu, ram)
        time.sleep(2)

threading.Thread(target=simulate_node_metrics, daemon=True).start()

# ==============================
# PHASE 48: Autonomous Cluster Scaling
# ==============================

import subprocess
import json

# ==============================
# 1. ClusterScaler
# ==============================
class ClusterScaler:
    def __init__(self, cluster_manager, predictor, rl_agent, logger, min_nodes=1, max_nodes=10):
        self.cluster_manager = cluster_manager
        self.predictor = predictor
        self.rl_agent = rl_agent
        self.logger = logger
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.active = True

    def run(self):
        while self.active:
            avg_cpu, avg_ram = self.calculate_cluster_load()
            self.logger.debug(f"[Scaler] Average cluster CPU={avg_cpu:.2f}, RAM={avg_ram:.2f}")
            if avg_cpu > 75 or avg_ram > 75:
                self.scale_up()
            elif avg_cpu < 30 and avg_ram < 30:
                self.scale_down()
            time.sleep(5)

    def calculate_cluster_load(self):
        cpus = []
        rams = []
        for node_id, node in self.cluster_manager.nodes.items():
            predicted_cpu, predicted_ram = self.predictor.predict_next(node_id)
            cpus.append(predicted_cpu)
            rams.append(predicted_ram)
        if not cpus:
            return 0, 0
        return np.mean(cpus), np.mean(rams)

    def scale_up(self):
        current_nodes = len(self.cluster_manager.nodes)
        if current_nodes >= self.max_nodes:
            self.logger.info("[Scaler] Max nodes reached, cannot scale up")
            return
        # Create a new node (simulate VM/container creation)
        new_node_id = f"node-{uuid.uuid4().hex[:6]}"
        new_node = ClusterNode(new_node_id)
        self.cluster_manager.register_node(new_node)
        self.logger.info(f"[Scaler] Scaled up: New node {new_node_id} added")
        # Optionally, start tasks on new node
        self.rl_agent.assign_tasks_to_new_node(new_node)

    def scale_down(self):
        current_nodes = len(self.cluster_manager.nodes)
        if current_nodes <= self.min_nodes:
            self.logger.info("[Scaler] Min nodes reached, cannot scale down")
            return
        # Identify candidate node to remove (lowest load)
        candidate_node = min(self.cluster_manager.nodes.values(), key=lambda n: np.mean(list(n.metrics.values())))
        if candidate_node.tasks_running:
            self.logger.debug(f"[Scaler] Node {candidate_node.node_id} still has tasks, skipping removal")
            return
        self.cluster_manager.nodes.pop(candidate_node.node_id, None)
        self.logger.info(f"[Scaler] Scaled down: Node {candidate_node.node_id} removed")

# ==============================
# 2. Integration with RL Agent
# ==============================
def assign_tasks_to_new_node(self, node):
    # Move pending tasks from overloaded nodes
    for task in list(self.rl_agent.pending_tasks):
        node.tasks_running[task.task_id] = task
        self.rl_agent.pending_tasks.remove(task)
        self.logger.info(f"[RLAgent] Assigned pending Task {task.task_id} to new Node {node.node_id}")

RLAgent.assign_tasks_to_new_node = assign_tasks_to_new_node

# ==============================
# 3. Phase 48 Bootstrap
# ==============================
cluster_scaler = ClusterScaler(cluster_manager, predictor, rl_agent, log_manager, min_nodes=1, max_nodes=10)
threading.Thread(target=cluster_scaler.run, daemon=True).start()
console.print("[ONYX] Phase 48: Autonomous Cluster Scaler online")

# ==============================
# 4. Example Node Metrics for Scaling Simulation
# ==============================
def simulate_scaling_metrics():
    while True:
        for node_id, node in cluster_manager.nodes.items():
            cpu = random.randint(10, 95)
            ram = random.randint(10, 95)
            node.metrics['cpu'] = cpu
            node.metrics['ram'] = ram
            predictor.record_metrics(node_id, cpu, ram)
        time.sleep(2)

threading.Thread(target=simulate_scaling_metrics, daemon=True).start()

# ==============================
# PHASE 49: Autonomous Network Adaptation
# ==============================

import socket
import shlex
import subprocess

# ==============================
# 1. NetworkAdapter
# ==============================
class NetworkAdapter:
    def __init__(self, traffic_analyzer, ids, vpn_manager, firewall_manager, logger):
        self.traffic_analyzer = traffic_analyzer
        self.ids = ids
        self.vpn_manager = vpn_manager
        self.firewall_manager = firewall_manager
        self.logger = logger
        self.active = True

    def run(self):
        while self.active:
            high_latency_hosts = self.detect_high_latency_hosts()
            suspicious_ips = self.ids.get_suspicious_ips()
            self.adjust_routing(high_latency_hosts)
            self.adjust_firewall(suspicious_ips)
            self.adjust_vpn_routes(high_latency_hosts)
            time.sleep(5)

    def detect_high_latency_hosts(self):
        high_latency = []
        for host in self.traffic_analyzer.monitored_hosts:
            latency = self.traffic_analyzer.ping_host(host)
            if latency > 200:  # milliseconds
                high_latency.append(host)
        self.logger.debug(f"[NetAdapter] High latency hosts: {high_latency}")
        return high_latency

    def adjust_routing(self, hosts):
        for host in hosts:
            # Example: Use alternative gateway or VPN
            self.logger.info(f"[NetAdapter] Adjusting route for {host}")
            # In real mode, run system commands or use netlink
            cmd = f"ip route replace {host} via 10.0.0.254"
            try:
                subprocess.run(shlex.split(cmd), check=True)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"[NetAdapter] Routing adjustment failed: {e}")

    def adjust_firewall(self, suspicious_ips):
        for ip in suspicious_ips:
            rule = f"deny from {ip}"
            try:
                self.firewall_manager.apply_rule(rule)
                self.logger.info(f"[NetAdapter] Firewall blocked suspicious IP: {ip}")
            except Exception as e:
                self.logger.error(f"[NetAdapter] Failed to block {ip}: {e}")

    def adjust_vpn_routes(self, hosts):
        for host in hosts:
            # Route high-latency host traffic through VPN
            try:
                self.vpn_manager.route_host(host)
                self.logger.info(f"[NetAdapter] Routed {host} through VPN")
            except Exception as e:
                self.logger.error(f"[NetAdapter] VPN routing failed for {host}: {e}")

# ==============================
# 2. Integration with FirewallManager
# ==============================
def apply_rule(self, rule):
    # Convert string rule to list for subprocess
    args = shlex.split(f"ufw {rule}")
    try:
        subprocess.run(args, check=True)
        self.logger.info(f"[Firewall] Applied rule: {rule}")
    except subprocess.CalledProcessError as e:
        self.logger.error(f"[Firewall] Failed to apply rule {rule}: {e}")

FirewallManager.apply_rule = apply_rule

# ==============================
# 3. Integration with VPNManager
# ==============================
def route_host(self, host):
    # Example: add a route for host via VPN interface
    cmd = f"ip route add {host}/32 dev {self.vpn_interface}"
    try:
        subprocess.run(shlex.split(cmd), check=True)
        self.logger.info(f"[VPN] Added route for {host} via {self.vpn_interface}")
    except subprocess.CalledProcessError as e:
        self.logger.error(f"[VPN] Failed to route {host}: {e}")

VPNManager.route_host = route_host

# ==============================
# 4. Phase 49 Bootstrap
# ==============================
network_adapter = NetworkAdapter(
    traffic_analyzer=traffic_analyzer,
    ids=intrusion_detection_system,
    vpn_manager=vpn_manager,
    firewall_manager=firewall_manager,
    logger=log_manager
)

threading.Thread(target=network_adapter.run, daemon=True).start()
console.print("[ONYX] Phase 49: Autonomous Network Adaptation online")

# ==============================
# 5. Example Traffic Simulation
# ==============================
def simulate_network_traffic():
    while True:
        for host in traffic_analyzer.monitored_hosts:
            latency = random.randint(50, 300)
            traffic_analyzer.latency_data[host] = latency
        time.sleep(2)

threading.Thread(target=simulate_network_traffic, daemon=True).start()

# ==============================
# PHASE 50: Autonomous Security Enforcement
# ==============================

import hashlib
import platform
import urllib.request
import tarfile
import json

# ==============================
# 1. SecurityEnforcer
# ==============================
class SecurityEnforcer:
    def __init__(self, ids, firewall_manager, vpn_manager, cluster_manager, logger, patch_repo_url):
        self.ids = ids
        self.firewall_manager = firewall_manager
        self.vpn_manager = vpn_manager
        self.cluster_manager = cluster_manager
        self.logger = logger
        self.patch_repo_url = patch_repo_url
        self.active = True
        self.patched_nodes = set()
        self.security_state_file = os.path.expanduser("~/.onyx/security_state.json")
        self.load_security_state()

    def load_security_state(self):
        if os.path.exists(self.security_state_file):
            with open(self.security_state_file, "r") as f:
                try:
                    self.patched_nodes = set(json.load(f).get("patched_nodes", []))
                except Exception as e:
                    self.logger.error(f"[SecurityEnforcer] Failed to load state: {e}")
                    self.patched_nodes = set()
        else:
            self.patched_nodes = set()

    def save_security_state(self):
        state = {"patched_nodes": list(self.patched_nodes)}
        with open(self.security_state_file, "w") as f:
            json.dump(state, f)
        self.logger.debug(f"[SecurityEnforcer] Security state saved")

    def run(self):
        while self.active:
            self.scan_and_block_suspicious_ips()
            self.apply_critical_patches()
            self.distribute_patches_to_cluster()
            time.sleep(10)

    def scan_and_block_suspicious_ips(self):
        suspicious_ips = self.ids.get_suspicious_ips()
        for ip in suspicious_ips:
            try:
                self.firewall_manager.apply_rule(f"deny from {ip}")
                self.logger.info(f"[SecurityEnforcer] Blocked malicious IP: {ip}")
            except Exception as e:
                self.logger.error(f"[SecurityEnforcer] Firewall block failed for {ip}: {e}")

    def apply_critical_patches(self):
        try:
            # Download patch manifest from repo
            manifest_url = f"{self.patch_repo_url}/latest_manifest.json"
            with urllib.request.urlopen(manifest_url) as response:
                manifest = json.loads(response.read().decode())
            for patch in manifest.get("critical_patches", []):
                patch_id = patch["id"]
                if patch_id in self.patched_nodes:
                    continue
                patch_url = f"{self.patch_repo_url}/{patch['file']}"
                local_patch_file = f"/tmp/{patch['file']}"
                urllib.request.urlretrieve(patch_url, local_patch_file)
                # Verify checksum
                with open(local_patch_file, "rb") as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
                if checksum != patch["sha256"]:
                    self.logger.error(f"[SecurityEnforcer] Patch {patch_id} failed checksum")
                    continue
                # Apply patch (tar extract + run install script)
                with tarfile.open(local_patch_file) as tar:
                    tar.extractall("/tmp/patch_apply")
                install_script = "/tmp/patch_apply/install.sh"
                subprocess.run(["bash", install_script], check=True)
                self.logger.info(f"[SecurityEnforcer] Applied patch {patch_id}")
                self.patched_nodes.add(patch_id)
                self.save_security_state()
        except Exception as e:
            self.logger.error(f"[SecurityEnforcer] Patch application error: {e}")

    def distribute_patches_to_cluster(self):
        # Send patch state to other nodes for cluster consistency
        for node in self.cluster_manager.nodes.values():
            if node["id"] not in self.patched_nodes:
                try:
                    self.cluster_manager.send_patch_state(node["id"], list(self.patched_nodes))
                    self.logger.info(f"[SecurityEnforcer] Distributed patches to node {node['id']}")
                except Exception as e:
                    self.logger.error(f"[SecurityEnforcer] Failed to distribute to {node['id']}: {e}")

# ==============================
# 2. Integration with ClusterManager
# ==============================
def send_patch_state(self, node_id, patch_list):
    try:
        # Simulated RPC call: in production, use ZeroMQ/gRPC
        self.logger.debug(f"[ClusterManager] Sent patch state to {node_id}: {patch_list}")
        # In real cluster: send JSON over TCP or pub-sub channel
    except Exception as e:
        self.logger.error(f"[ClusterManager] Failed to send patch state to {node_id}: {e}")

ClusterManager.send_patch_state = send_patch_state

# ==============================
# 3. Phase 50 Bootstrap
# ==============================
security_enforcer = SecurityEnforcer(
    ids=intrusion_detection_system,
    firewall_manager=firewall_manager,
    vpn_manager=vpn_manager,
    cluster_manager=cluster_manager,
    logger=log_manager,
    patch_repo_url="https://onyx-patches.example.com"
)

threading.Thread(target=security_enforcer.run, daemon=True).start()
console.print("[ONYX] Phase 50: Autonomous Security Enforcement online")

# ==============================
# 4. Security Simulation
# ==============================
def simulate_ids_alerts():
    while True:
        ip = f"192.168.1.{random.randint(2, 254)}"
        intrusion_detection_system.alert_ip(ip)
        time.sleep(3)

threading.Thread(target=simulate_ids_alerts, daemon=True).start()



