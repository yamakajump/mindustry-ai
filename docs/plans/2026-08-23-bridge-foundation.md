# Bridge Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pinned Mindustry headless server running our own plugin, scriptable from Python, with a measured answer to the question that governs the whole project: how many game ticks per second can one instance actually produce?

**Architecture:** A Gradle-built JVM plugin (`mindustry.mod.Plugin`) is dropped into a headless Mindustry server. A Python harness starts that server as a subprocess, drives it through stdin, and reads its stdout. Every test in this plan is an integration test against a real server, because Mindustry state only exists inside a running engine and mocking it would test nothing.

**Tech Stack:** Java 17, Gradle 9.4.1, Mindustry v159.7 (pinned), Python 3.13, pytest.

## Global Constraints

- **Java 17 exactly.** Mindustry targets `JavaVersion.VERSION_17`. Newer JDKs are not validated.
- **Mindustry pinned to `v159.7`.** Never `latest`. A silent engine bump invalidates every measurement in the repository.
- **Code, comments, identifiers and documentation in English.** Commit messages in French, conventional format.
- **No em dash anywhere**, in any file. Use a comma, a colon, or a short dash.
- **The bridge contains no strategy.** If a change makes the plugin decide what a good move is, it belongs in Python instead. See `docs/decisions/0002-three-process-architecture.md`.
- **Every performance number must come from a measurement**, never an estimate, and must state the hardware it was measured on.
- **Work on a feature branch.** `feat/bridge-foundation`, PR into `main`, squash merge.

## File Structure

| File | Responsibility |
|---|---|
| `bridge/build.gradle` | Plugin build, pinned Mindustry dependency, jar packaging |
| `bridge/settings.gradle` | Gradle project name |
| `bridge/plugin.json` | Plugin manifest read by Mindustry at load time |
| `bridge/src/mindustryai/BridgePlugin.java` | Plugin entry point, command registration only |
| `bridge/src/mindustryai/Clock.java` | Simulation speed control |
| `bridge/src/mindustryai/Benchmark.java` | Tick throughput measurement and reporting |
| `tools/mindustry_server.py` | Python harness: download, launch, drive, tear down a server |
| `tools/setup_server.py` | One-time server provisioning, idempotent |
| `tests/test_server_harness.py` | The harness itself works |
| `tests/test_plugin_loads.py` | The plugin loads and answers |
| `tests/test_benchmark.py` | The benchmark produces a plausible number |
| `docs/measurements/throughput.md` | Recorded results, with hardware |

Rationale for the split: `BridgePlugin` stays a thin registration surface so that later work (observations, actions, masks, events) adds files rather than growing one. `Clock` and `Benchmark` are separable because the first is a capability and the second is a diagnostic.

---

### Task 1: Server harness

**Files:**
- Create: `pyproject.toml`
- Create: `tools/__init__.py` (empty)
- Create: `tools/setup_server.py`
- Create: `tools/mindustry_server.py`
- Test: `tests/test_server_harness.py`

**Interfaces:**
- Produces: `setup_server(root: Path, version: str = "v159.7") -> Path` returning the server directory. `ServerProcess(server_dir: Path)` as a context manager, with `send(command: str) -> None`, `wait_for(pattern: str, timeout: float = 30.0) -> str` returning the matched line, and `lines() -> list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server_harness.py
from pathlib import Path
import pytest
from tools.setup_server import setup_server
from tools.mindustry_server import ServerProcess


@pytest.fixture(scope="session")
def server_dir(tmp_path_factory) -> Path:
    return setup_server(tmp_path_factory.mktemp("mindustry"))


def test_setup_downloads_pinned_server_jar(server_dir: Path):
    jar = server_dir / "server-release.jar"
    assert jar.exists()
    # v159.7 server jar is ~18.3 MB. Guard against a truncated download.
    assert jar.stat().st_size > 15_000_000


def test_server_starts_and_reports_version(server_dir: Path):
    with ServerProcess(server_dir) as server:
        server.send("version")
        line = server.wait_for(r"Server Version")
        assert "159" in line


def test_server_hosts_a_map(server_dir: Path):
    with ServerProcess(server_dir) as server:
        server.send("host Ancient_Caldera survival")
        line = server.wait_for(r"Opened a server", timeout=60)
        assert line
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m pytest tests/test_server_harness.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'tools.setup_server'`

- [ ] **Step 3: Make `tools` importable from the tests**

Without this, every `from tools.setup_server import ...` fails with `ModuleNotFoundError`,
whatever the working directory.

```toml
# pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

Also create an empty `tools/__init__.py`.

- [ ] **Step 4: Implement the provisioning script**

```python
# tools/setup_server.py
"""Download and lay out a pinned Mindustry headless server."""
from pathlib import Path
import urllib.request

MINDUSTRY_VERSION = "v159.7"
SERVER_JAR_URL = (
    "https://github.com/Anuken/Mindustry/releases/download/{version}/server-release.jar"
)


def setup_server(root: Path, version: str = MINDUSTRY_VERSION) -> Path:
    """Provision a server directory. Idempotent: an existing jar is left alone."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config" / "mods").mkdir(parents=True, exist_ok=True)

    jar = root / "server-release.jar"
    if not jar.exists():
        url = SERVER_JAR_URL.format(version=version)
        tmp = jar.with_suffix(".jar.part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(jar)

    return root


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mindustry-server")
    print(setup_server(target))
```

- [ ] **Step 5: Implement the process harness**

The server reads commands from stdin and writes to stdout. Reading stdout must happen on a
background thread, otherwise a full pipe buffer deadlocks the server.

```python
# tools/mindustry_server.py
"""Drive a Mindustry headless server as a subprocess."""
from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path


class ServerProcess:
    """Context manager wrapping a running headless server."""

    def __init__(self, server_dir: Path, java: str = "java", extra_args: list[str] | None = None):
        self.server_dir = server_dir
        self.java = java
        self.extra_args = extra_args or []
        self._proc: subprocess.Popen[str] | None = None
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def __enter__(self) -> "ServerProcess":
        self._proc = subprocess.Popen(
            [self.java, *self.extra_args, "-jar", "server-release.jar"],
            cwd=self.server_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        self.wait_for(r"Server loaded\.|Loaded \d+ mod", timeout=60)
        return self

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            with self._lock:
                self._lines.append(line.rstrip("\n"))

    def send(self, command: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def lines(self) -> list[str]:
        with self._lock:
            return list(self._lines)

    def wait_for(self, pattern: str, timeout: float = 30.0) -> str:
        """Block until a line matches, and return it. Raises TimeoutError otherwise."""
        deadline = time.monotonic() + timeout
        compiled = re.compile(pattern)
        seen = 0
        while time.monotonic() < deadline:
            current = self.lines()
            for line in current[seen:]:
                if compiled.search(line):
                    return line
            seen = len(current)
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self._proc.returncode}\n"
                    + "\n".join(current[-40:])
                )
            time.sleep(0.05)
        raise TimeoutError(
            f"no line matched {pattern!r} within {timeout}s\n"
            + "\n".join(self.lines()[-40:])
        )

    def __exit__(self, *exc_info: object) -> None:
        if self._proc is None:
            return
        try:
            self.send("exit")
            self._proc.wait(timeout=15)
        except Exception:
            self._proc.kill()
            self._proc.wait(timeout=10)
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_server_harness.py -v`
Expected: PASS, 3 tests.

If `test_server_hosts_a_map` fails on the map name, list what the build actually ships by
running `maps` at the server prompt and adjust the test to a name that exists. Do not
weaken the assertion to make it pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tools/__init__.py tools/setup_server.py tools/mindustry_server.py tests/test_server_harness.py
git commit -m "feat(tools): harnais Python pour piloter un serveur Mindustry headless

Provisionne un serveur epingle sur v159.7 et l'expose comme context
manager pilotable par stdin. La lecture de stdout passe par un thread
dedie : un pipe plein bloquerait le serveur.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Plugin that loads and answers

**Files:**
- Create: `bridge/build.gradle`
- Create: `bridge/settings.gradle`
- Create: `bridge/plugin.json`
- Create: `bridge/src/mindustryai/BridgePlugin.java`
- Modify: `tools/mindustry_server.py` (add `install_plugin`)
- Test: `tests/test_plugin_loads.py`

**Interfaces:**
- Consumes: `ServerProcess`, `setup_server` from Task 1.
- Produces: `install_plugin(server_dir: Path, jar: Path) -> None`. Server command `bridge-status` printing `bridge ready version=<v>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plugin_loads.py
import subprocess
from pathlib import Path
import pytest
from tools.setup_server import setup_server
from tools.mindustry_server import ServerProcess, install_plugin

BRIDGE = Path(__file__).parent.parent / "bridge"


@pytest.fixture(scope="session")
def server_with_plugin(tmp_path_factory) -> Path:
    gradlew = "gradlew.bat" if __import__("os").name == "nt" else "./gradlew"
    subprocess.run([gradlew, "jar"], cwd=BRIDGE, check=True)
    jar = next((BRIDGE / "build" / "libs").glob("*.jar"))
    server_dir = setup_server(tmp_path_factory.mktemp("mindustry"))
    install_plugin(server_dir, jar)
    return server_dir


def test_plugin_is_loaded(server_with_plugin: Path):
    with ServerProcess(server_with_plugin) as server:
        server.send("mods")
        line = server.wait_for(r"mindustry-ai-bridge")
        assert line


def test_bridge_status_command_answers(server_with_plugin: Path):
    with ServerProcess(server_with_plugin) as server:
        server.send("bridge-status")
        line = server.wait_for(r"bridge ready")
        assert "version=" in line
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m pytest tests/test_plugin_loads.py -v`
Expected: FAIL, `ImportError: cannot import name 'install_plugin'`

- [ ] **Step 3: Write the Gradle build**

Mirrors `Anuken/MindustryPluginTemplate`, with the version pinned instead of `latest`.

```groovy
// bridge/build.gradle
apply plugin: "java"

sourceSets.main.java.srcDirs = ["src"]

repositories {
    mavenCentral()

    // Mindustry publishes dependency jars as release artifacts rather than to a
    // real repository. An Ivy layout pointed at GitHub is the supported approach.
    ivy {
        url = "https://github.com/"
        patternLayout { artifact "/[organisation]/[module]/releases/download/[revision]/dependencies.jar" }
        metadataSources { artifact() }
    }
}

java {
    targetCompatibility = JavaVersion.VERSION_17
    sourceCompatibility = JavaVersion.VERSION_17
}

ext {
    // Pinned deliberately. Never "latest": a silent engine bump would
    // invalidate every measurement recorded in this repository.
    mindustryVersion = "v159.7"
}

dependencies {
    compileOnly "Anuken:Mindustry:$mindustryVersion"
}

jar {
    archiveFileName = "mindustry-ai-bridge.jar"
    from(rootDir) { include "plugin.json" }
}
```

```groovy
// bridge/settings.gradle
rootProject.name = "mindustry-ai-bridge"
```

```json
{
  "name": "mindustry-ai-bridge",
  "author": "yamakajump",
  "main": "mindustryai.BridgePlugin",
  "description": "Exposes Mindustry state and actions to external learning agents.",
  "version": "0.1.0",
  "minGameVersion": "146"
}
```

- [ ] **Step 4: Write the plugin entry point**

```java
// bridge/src/mindustryai/BridgePlugin.java
package mindustryai;

import arc.util.CommandHandler;
import arc.util.Log;
import mindustry.mod.Plugin;

/**
 * Entry point for the mindustry-ai bridge.
 *
 * <p>This class registers commands and owns nothing else. It deliberately contains no
 * game logic and no strategy: anything resembling a decision about what constitutes a
 * good move belongs in the Python side, where it can be learned rather than hardcoded.
 */
public class BridgePlugin extends Plugin {
    public static final String VERSION = "0.1.0";

    @Override
    public void init() {
        Log.info("[mindustry-ai] bridge @ loaded", VERSION);
    }

    @Override
    public void registerServerCommands(CommandHandler handler) {
        handler.register("bridge-status", "Report bridge version and current tick rate.", args ->
            Log.info("bridge ready version=@", VERSION)
        );
    }
}
```

- [ ] **Step 5: Add plugin installation to the harness**

```python
# append to tools/mindustry_server.py
import shutil


def install_plugin(server_dir: Path, jar: Path) -> None:
    """Copy a built plugin jar into the server mod directory."""
    mods = server_dir / "config" / "mods"
    mods.mkdir(parents=True, exist_ok=True)
    shutil.copy2(jar, mods / jar.name)
```

- [ ] **Step 6: Set up the Gradle wrapper**

Run, from `bridge/`:

```bash
gradle wrapper --gradle-version 9.4.1
```

If no system Gradle is available, copy `gradlew`, `gradlew.bat` and `gradle/wrapper/`
from `https://github.com/Anuken/MindustryPluginTemplate` and edit `distributionUrl` in
`gradle/wrapper/gradle-wrapper.properties` to `gradle-9.4.1-bin.zip`.

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_plugin_loads.py -v`
Expected: PASS, 2 tests.

`arc.Core.graphics` may be null in a headless build. If `bridge-status` throws a
`NullPointerException`, report `-1` for tps at this stage and leave a comment saying why.
Task 4 replaces this field with a real measurement regardless.

- [ ] **Step 8: Commit**

```bash
git add bridge/ tools/mindustry_server.py tests/test_plugin_loads.py
git commit -m "feat(bridge): squelette du plugin JVM charge par le serveur

Build Gradle epingle sur Mindustry v159.7, manifeste plugin.json, et une
commande bridge-status qui prouve que le plugin repond. Le point d'entree
reste une simple surface d'enregistrement : aucune logique de jeu.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Tick counter at default speed

**Files:**
- Create: `bridge/src/mindustryai/Benchmark.java`
- Modify: `bridge/src/mindustryai/BridgePlugin.java`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `ServerProcess`, `install_plugin`.
- Produces: server command `bridge-bench <seconds>` printing exactly one line, `bench ticks=<n> seconds=<s> tps=<f>`.

Mindustry targets 60 ticks per second. At default speed this benchmark must report close
to 60. That is the point: a number near 60 proves the counter is correct, which is what
makes the accelerated number in Task 5 trustworthy.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
import re
from pathlib import Path
from tools.mindustry_server import ServerProcess

# server_with_plugin fixture is reused from tests/test_plugin_loads.py via conftest.py


def test_benchmark_reports_about_sixty_tps_at_default_speed(server_with_plugin: Path):
    with ServerProcess(server_with_plugin) as server:
        server.send("host Ancient_Caldera survival")
        server.wait_for(r"Opened a server", timeout=60)
        server.send("bridge-bench 5")
        line = server.wait_for(r"bench ticks=", timeout=30)

        tps = float(re.search(r"tps=([\d.]+)", line).group(1))
        # Mindustry targets 60 TPS. Anything far from that means the counter is wrong,
        # not that the machine is slow.
        assert 50.0 <= tps <= 70.0, f"unexpected default tick rate: {line}"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: FAIL, TimeoutError, no line matching `bench ticks=`

- [ ] **Step 3: Implement the tick counter**

```java
// bridge/src/mindustryai/Benchmark.java
package mindustryai;

import arc.Core;
import arc.Events;
import arc.util.Log;
import mindustry.game.EventType.Trigger;

/**
 * Measures how many game ticks the server actually produces per wall-clock second.
 *
 * <p>This number governs what is trainable at all, so it is measured rather than
 * assumed, and reported with the wall-clock window it was measured over.
 */
public class Benchmark {
    private long ticks;
    private boolean running;
    private long startNanos;
    private long targetNanos;

    public Benchmark() {
        Events.run(Trigger.update, this::onTick);
    }

    private void onTick() {
        if (!running) {
            return;
        }
        ticks++;
        long elapsed = System.nanoTime() - startNanos;
        if (elapsed >= targetNanos) {
            running = false;
            double seconds = elapsed / 1_000_000_000.0;
            Log.info("bench ticks=@ seconds=@ tps=@",
                ticks, String.format("%.3f", seconds),
                String.format("%.2f", ticks / seconds));
        }
    }

    /** Begin a measurement window. Results are logged when it closes. */
    public void start(int seconds) {
        if (running) {
            Log.warn("bench already running");
            return;
        }
        this.ticks = 0;
        this.startNanos = System.nanoTime();
        this.targetNanos = seconds * 1_000_000_000L;
        this.running = true;
        Log.info("bench started window=@s", seconds);
    }
}
```

- [ ] **Step 4: Wire the command in**

Replace the body of `registerServerCommands` in `BridgePlugin.java`:

```java
    private final Benchmark benchmark = new Benchmark();

    @Override
    public void registerServerCommands(CommandHandler handler) {
        handler.register("bridge-status", "Report bridge version.", args ->
            Log.info("bridge ready version=@", VERSION)
        );

        handler.register("bridge-bench", "<seconds>", "Measure tick throughput.", args -> {
            int seconds = Integer.parseInt(args[0]);
            benchmark.start(seconds);
        });
    }
```

Note that `benchmark` is a field initialised at construction, so its `Trigger.update`
listener is registered before any command runs.

- [ ] **Step 5: Add the shared fixture**

```python
# tests/conftest.py
import os
import subprocess
from pathlib import Path
import pytest
from tools.setup_server import setup_server
from tools.mindustry_server import install_plugin

BRIDGE = Path(__file__).parent.parent / "bridge"


@pytest.fixture(scope="session")
def server_with_plugin(tmp_path_factory) -> Path:
    gradlew = "gradlew.bat" if os.name == "nt" else "./gradlew"
    subprocess.run([gradlew, "jar"], cwd=BRIDGE, check=True)
    jar = next((BRIDGE / "build" / "libs").glob("*.jar"))
    server_dir = setup_server(tmp_path_factory.mktemp("mindustry"))
    install_plugin(server_dir, jar)
    return server_dir
```

Remove the duplicated fixture from `tests/test_plugin_loads.py`.

- [ ] **Step 6: Run the full suite and confirm it passes**

Run: `python -m pytest tests/ -v`
Expected: PASS, all tests.

- [ ] **Step 7: Commit**

```bash
git add bridge/src/mindustryai/ tests/
git commit -m "feat(bridge): compteur de ticks et commande bridge-bench

Mesure le debit reel en ticks par seconde sur une fenetre donnee. A vitesse
par defaut le resultat doit approcher 60 : c'est ce qui prouve que le
compteur est juste, donc que la mesure acceleree sera credible.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Simulation acceleration

**Files:**
- Create: `bridge/src/mindustryai/Clock.java`
- Modify: `bridge/src/mindustryai/BridgePlugin.java`
- Test: `tests/test_clock.py`

**Interfaces:**
- Produces: server command `bridge-speed <n|max>`. `Clock.setMultiplier(int)` where 1 is realtime and `max` runs uncapped.

**This task is an investigation, not a transcription.** How to accelerate a headless
Mindustry server from inside a plugin is not established, so the work is to test three
hypotheses in order and keep the first that holds. Write down which one won and why, in
`docs/measurements/throughput.md`.

Do not skip ahead to a hypothesis that looks more elegant. They are ordered by risk to
simulation correctness, cheapest first.

**Hypothesis A: extra logic updates per frame.** Register an `ApplicationListener` that
calls `Vars.logic.update()` an extra `n - 1` times per frame. Read
`Mindustry/core/src/mindustry/core/Logic.java` first to see what `update()` touches.
Risk: anything driven by frame count rather than by `Time.delta` will run at the wrong
rate, and net code may fight it.

**Hypothesis B: raise `Time.delta`.** Scale the per-frame delta so each tick advances more
game time. Risk: high. Physics, movement and collision assume small deltas, so units may
tunnel through walls. Verify with a real match, not with the tick counter alone.

**Hypothesis C: shorten the headless render interval.** `arc.backend.headless.HeadlessApplication`
takes a `renderIntervalSec` in its constructor, and a value of `0` runs uncapped. The
server sets this in `ServerLauncher`, before a plugin loads, so reaching it from a plugin
requires reflection or a custom launcher. Cleanest in principle, most invasive in practice.

- [ ] **Step 1: Get the engine source for reference**

```bash
git clone --depth 1 --branch v159.7 https://github.com/Anuken/Mindustry.git ../Mindustry-reference
```

Read, in this order: `core/src/mindustry/core/Logic.java`, `server/src/mindustry/server/ServerLauncher.java`,
and `arc/backends/backend-headless/src/arc/backend/headless/HeadlessApplication.java` from the Arc repository.

Do not add this clone to the repository. It is reference material, and `.gitignore`
already excludes `mindustry-server/`; add `Mindustry-reference/` to it as well.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_clock.py
import re
from pathlib import Path
from tools.mindustry_server import ServerProcess


def _bench(server: ServerProcess, seconds: int = 5) -> float:
    server.send(f"bridge-bench {seconds}")
    line = server.wait_for(r"bench ticks=", timeout=seconds * 6 + 30)
    return float(re.search(r"tps=([\d.]+)", line).group(1))


def test_acceleration_multiplies_tick_rate(server_with_plugin: Path):
    with ServerProcess(server_with_plugin) as server:
        server.send("host Ancient_Caldera survival")
        server.wait_for(r"Opened a server", timeout=60)

        baseline = _bench(server)
        server.send("bridge-speed 8")
        server.wait_for(r"speed set")
        accelerated = _bench(server)

        # A conservative floor: 8x requested, 4x accepted as proof the mechanism works.
        # The exact ceiling is hardware-bound and is recorded, not asserted.
        assert accelerated > baseline * 4, f"{baseline=} {accelerated=}"
```

- [ ] **Step 3: Run the test and confirm it fails**

Run: `python -m pytest tests/test_clock.py -v`
Expected: FAIL, TimeoutError waiting for `speed set`

- [ ] **Step 4: Implement Hypothesis A**

```java
// bridge/src/mindustryai/Clock.java
package mindustryai;

import arc.ApplicationListener;
import arc.Core;
import arc.util.Log;
import mindustry.Vars;

/**
 * Controls how fast the simulation advances relative to wall-clock time.
 *
 * <p>Runs extra logic updates per frame rather than inflating Time.delta, because
 * movement and collision code assumes small deltas and misbehaves when they grow.
 */
public class Clock implements ApplicationListener {
    private int multiplier = 1;

    public void install() {
        Core.app.addListener(this);
    }

    /** 1 is realtime. Higher values run that many logic updates per frame. */
    public void setMultiplier(int multiplier) {
        this.multiplier = Math.max(1, multiplier);
        Log.info("speed set multiplier=@", this.multiplier);
    }

    @Override
    public void update() {
        // The engine already ran one update this frame. Add the remainder.
        for (int i = 1; i < multiplier; i++) {
            Vars.logic.update();
        }
    }
}
```

Wire it in `BridgePlugin.init()`:

```java
    private final Clock clock = new Clock();

    @Override
    public void init() {
        Log.info("[mindustry-ai] bridge @ loaded", VERSION);
        clock.install();
    }
```

And register the command:

```java
        handler.register("bridge-speed", "<n|max>", "Set simulation speed multiplier.", args -> {
            int n = args[0].equals("max") ? 1000 : Integer.parseInt(args[0]);
            clock.setMultiplier(n);
        });
```

- [ ] **Step 5: Run the test**

Run: `python -m pytest tests/test_clock.py -v`

If it passes, go to Step 6. If the tick rate does not move, the listener is being added
after the server built its listener list; try `Core.app.post(() -> Core.app.addListener(this))`
in `install()`. If it still does not move, move to Hypothesis B, then C. Record every
attempt, including the ones that failed.

- [ ] **Step 6: Verify the simulation is still correct**

Acceleration that corrupts the game is worthless. Run a real match under acceleration and
confirm the world still behaves.

```bash
python -m pytest tests/test_clock.py -v
```

Then manually, at the server prompt:

```
host Ancient_Caldera survival
bridge-speed 8
```

Wait for several waves, then run `status`. Enemies must spawn, units must move, and the
core must take damage over time. If units freeze, pass through walls, or waves stop
arriving, the hypothesis has broken the simulation: reject it and move to the next one,
however good its throughput number looked.

- [ ] **Step 7: Record the measurement**

```markdown
<!-- docs/measurements/throughput.md -->
# Throughput measurements

Every number here is measured, never estimated, and states its hardware.

## Reference hardware

Ryzen 9 9950X3D (32 threads), RTX 5090 32 GB, 93 GB RAM, Windows 11.

## Single instance, Ancient_Caldera, survival

| Speed setting | Ticks per second | Ratio to realtime | Date |
|---|---|---|---|
| 1 (default) | _fill in_ | 1.0x | 2026-08-23 |
| 8 | _fill in_ | _fill in_ | 2026-08-23 |
| max | _fill in_ | _fill in_ | 2026-08-23 |

## Method

Which hypothesis won, which were rejected, and what broke in the rejected ones.

## What this implies

Episodes per hour for a ten-minute game, at the measured rate, times 24 concurrent
instances. This figure decides whether the curriculum in `docs/architecture.md` is
trainable as designed.
```

Fill in every `_fill in_` with a real number before committing. A table of blanks is worse
than no table.

- [ ] **Step 8: Commit**

```bash
git add bridge/src/mindustryai/Clock.java bridge/src/mindustryai/BridgePlugin.java tests/test_clock.py docs/measurements/throughput.md .gitignore
git commit -m "feat(bridge): controle de la vitesse de simulation

Accelere la simulation par updates logiques supplementaires par frame,
plutot qu'en gonflant Time.delta qui casserait les collisions. Consigne
le debit mesure et les hypotheses ecartees.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Continuous integration

**Files:**
- Create: `.github/workflows/bridge.yml`
- Create: `bridge/.gitignore`

**Interfaces:**
- Consumes: everything above.

CI runs the Java build and the fast tests on every push. It deliberately does not run the
throughput benchmark: a shared runner produces a number that means nothing, and publishing
a meaningless number is worse than publishing none.

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/bridge.yml
name: bridge

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-java@v4
        with:
          distribution: temurin
          java-version: "17"

      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"

      - name: Build the plugin
        working-directory: bridge
        run: ./gradlew jar --no-daemon

      - name: Install test dependencies
        run: pip install pytest

      - name: Run integration tests
        # Excludes the throughput benchmark: a shared runner cannot produce a
        # meaningful performance number, and a meaningless one is worse than none.
        run: python -m pytest tests/ -v --deselect tests/test_clock.py
```

- [ ] **Step 2: Verify the workflow passes**

```bash
git push -u origin feat/bridge-foundation
gh pr create --fill
gh run watch
```

Expected: the `build` job succeeds. If the Gradle step fails on a missing wrapper, the
wrapper from Task 2 Step 6 was not committed. Check with `git ls-files bridge/gradle`.

- [ ] **Step 3: Commit and open the pull request**

```bash
git add .github/workflows/bridge.yml bridge/.gitignore
git commit -m "ci: build du plugin et tests d'integration sur chaque push

Le benchmark de debit est exclu volontairement : un runner partage ne
produit aucun chiffre exploitable, et publier un chiffre faux est pire
que ne rien publier.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
git push
```

---

## Definition of done

This plan is complete when all of the following hold:

- `python -m pytest tests/ -v` passes from a clean checkout, on a machine with only Java 17 and Python 3.13 installed.
- `docs/measurements/throughput.md` contains real numbers, with the hardware stated and no blanks.
- The accelerated simulation has been verified to still play correctly, not merely to tick faster.
- CI is green on the pull request.
- The README roadmap has `Bridge` and `Throughput benchmark` checked.

## What this plan deliberately does not build

Recorded so the boundary is explicit, and so the next plan knows where it starts.

- No socket protocol. The bridge is driven by server commands only, which is enough to prove the foundation works.
- No observation encoding, no tensors.
- No action execution, no legality masks.
- No event log, no replays.

Each of those is its own plan, and each depends on the throughput number this one produces.
