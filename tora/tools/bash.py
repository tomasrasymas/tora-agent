"""Bash tool: run commands on the host system.

This gives the agent general access to the machine TORA runs on — reading and
writing files, inspecting the system, running programs — all through bash.
It is powerful and runs with the privileges of the TORA process, so only enable
it on a machine you trust the model to operate.

As a guardrail (not a sandbox), commands are parsed into an AST with `bashlex`
and rejected if any command — including ones nested in pipes, `&&`/`||`,
subshells, or `$(...)`/backtick substitutions — names a known-dangerous binary
(see `BLOCKED_COMMANDS`). `eval` is blocked because its string argument is
opaque to the parser and would otherwise be a trivial bypass. Commands that
bashlex cannot parse are rejected (fail closed). This stops accidents and naive
footguns; it is not a security boundary against a determined adversary.
"""

import os
import subprocess

import bashlex
from bashlex import ast as bashlex_ast
from bashlex import errors as bashlex_errors

from .base import Tool

# Keep captured output from overwhelming the model's context.
_MAX_OUTPUT_CHARS = 10_000

# Commands the agent is never allowed to run. Matched on the binary's basename,
# so `/bin/rm` is caught too. Edit this set to tighten or loosen the guardrail.
BLOCKED_COMMANDS = frozenset(
    {
        # Destroying data / filesystems
        "rm",
        "rmdir",
        "shred",
        "dd",
        "mkfs",  # plus any mkfs.* variant (see _is_blocked)
        "mkswap",
        "fdisk",
        "parted",
        "wipefs",
        "blkdiscard",
        # Mounting / device fiddling
        "mount",
        "umount",
        # Killing processes
        "kill",
        "killall",
        "pkill",
        # Power / runlevel
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "init",
        "telinit",
        # Opaque indirection that would bypass AST checks
        "eval",
    }
)


def _is_blocked(name: str) -> bool:
    """True if a command basename is on the blocklist."""
    base = os.path.basename(name)
    return base in BLOCKED_COMMANDS or base.startswith("mkfs.")


class _CommandCollector(bashlex_ast.nodevisitor):
    """Collects the literal command name from every command node in the AST.

    The visitor recurses into substitutions, so commands hidden inside `$(...)`
    or backticks are caught too. Words that themselves contain expansions (e.g.
    the outer `` `reboot` `` token) are skipped — their inner command is visited
    on its own, and the wrapped token isn't a real binary name.
    """

    def __init__(self) -> None:
        self.names: list[str] = []

    def visitcommand(self, n, parts) -> None:
        for part in parts:
            if part.kind == "word" and not part.parts:
                self.names.append(part.word)
                break  # first literal word is the command name


def _blocked_command(command: str) -> str | None:
    """Return the name of the first blocked command, or None if all are allowed.

    Raises bashlex parsing errors to the caller so they can fail closed.
    """
    collector = _CommandCollector()
    for tree in bashlex.parse(command):
        collector.visit(tree)
    return next((name for name in collector.names if _is_blocked(name)), None)


def _clip(text: str) -> str:
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    dropped = len(text) - _MAX_OUTPUT_CHARS
    return text[:_MAX_OUTPUT_CHARS] + f"\n…[truncated {dropped} chars]"


def bash(command: str, timeout: int = 60) -> dict:
    """Run a bash command and capture its result.

    Returns the exit code plus (clipped) stdout and stderr. Errors — a blocked
    command, a syntax the parser can't handle, a timeout, or a failure to
    launch — come back as a result rather than an exception, so the model can
    read what went wrong and try again.
    """

    try:
        blocked = _blocked_command(command)
    except bashlex_errors.ParsingError as e:
        # Fail closed: if we can't understand the command, we don't run it.
        return {
            "error": f"Could not parse command, refusing to run: {e}",
            "command": command,
        }
    if blocked is not None:
        return {
            "error": f"Command '{blocked}' is blocked for safety.",
            "command": command,
        }

    try:
        proc = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",  # so bashisms work; sh (dash) would not
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


bash_tool = Tool(
    name="bash",
    description=(
        "Run a bash command on the host machine and return its exit code, "
        "stdout, and stderr. Use this to read and write files, inspect the "
        "system, and run programs (e.g. `cat ~/training.csv`, "
        "`echo ... > notes.md`, `ls -la`). Combine steps with `&&` and change "
        "directory inside the command when needed (e.g. `cd ~/logs && ls`). "
        "Prefer `rg` (ripgrep) over `grep` for searching files — it is much "
        "faster (e.g. `rg -n TODO ~/notes`). "
        "Destructive or system-altering commands (e.g. rm, dd, mkfs, kill, "
        "shutdown) are blocked and will return an error."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait before the command is killed (default 60).",
            },
        },
        "required": ["command"],
    },
    fn=bash,
)
