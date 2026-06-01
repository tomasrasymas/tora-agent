"""Shell tool: run commands on the host system.

This gives the agent general access to the machine TORA runs on — reading and
writing files, inspecting the system, running programs — all through the shell.
It is powerful and unsandboxed: every command runs with the privileges of the
TORA process, so only enable it on a machine you trust the model to operate.
"""

import subprocess

from .base import Tool

# Keep captured output from overwhelming the model's context.
_MAX_OUTPUT_CHARS = 10_000


def _clip(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    dropped = len(text) - _MAX_OUTPUT_CHARS
    return text[:_MAX_OUTPUT_CHARS] + f"\n…[truncated {dropped} chars]"


def run_shell(command: str, timeout: int = 60) -> dict:
    """Run a shell command and capture its result.

    Returns the exit code plus (clipped) stdout and stderr. Errors — a timeout
    or a failure to launch — come back as a result rather than an exception, so
    the model can read what went wrong and try again.
    """

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s", "command": command}
    except Exception as e:
        return {"error": str(e), "command": command}

    return {
        "exit_code": proc.returncode,
        "stdout": _clip(proc.stdout),
        "stderr": _clip(proc.stderr),
    }


shell_tool = Tool(
    name="run_shell",
    description=(
        "Run a shell command on the host machine and return its exit code, "
        "stdout, and stderr. Use this to read and write files, inspect the "
        "system, and run programs (e.g. `cat ~/training.csv`, "
        "`echo ... > notes.md`, `ls -la`). Combine steps with `&&` and change "
        "directory inside the command when needed (e.g. `cd ~/logs && ls`)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait before the command is killed (default 60).",
            },
        },
        "required": ["command"],
    },
    fn=run_shell,
)
