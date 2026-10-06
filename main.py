"""claude-task action.

Runs a headless Claude Code agent (`claude -p`) to carry out a free-form
task, streaming progress into the step logs and reporting the agent's
final message as the `report` output.

Uses only the Python standard library. The claude CLI must exist on the
runner (PATH or ~/.local/bin) and be authenticated (prior `claude login`
on the machine, or ANTHROPIC_API_KEY in the runner's environment).

Inputs arrive as a single JSON string in sys.argv[-1]. Outputs are
reported as `::key::<json>` stdout lines.

Cancellation: the agent kills this process's subprocess tree on step
cancel/timeout, which terminates the claude subprocess with it.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

# Pinned rather than the CLI default, which differs per runner machine.
DEFAULT_MODEL = "claude-opus-5-5"


def find_claude() -> str:
    found = shutil.which("claude")
    if found:
        return found
    fallback = os.path.expanduser("~/.local/bin/claude")
    if os.path.exists(fallback):
        return fallback
    print(
        "claude CLI not found on PATH or in ~/.local/bin. "
        "Install it on the runner: curl -fsSL https://claude.ai/install.sh | bash",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    inputs = json.loads(sys.argv[-1])

    task = inputs.get("task")
    if not task:
        print("Input 'task' is required.", file=sys.stderr)
        sys.exit(1)

    allowed_tools = (
        inputs.get("allowed_tools")
        or "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch"
    )
    max_turns = int(inputs.get("max_turns") or 0)
    model = inputs.get("model") or DEFAULT_MODEL

    cmd = [
        find_claude(),
        "-p",
        task,
        "--output-format",
        "stream-json",
        "--verbose",
        "--allowedTools",
        allowed_tools,
    ]
    if max_turns > 0:
        cmd += ["--max-turns", str(max_turns)]
    cmd += ["--model", model]

    # Fresh working directory per run so tasks never see each other's files.
    workdir = tempfile.mkdtemp(prefix="claude-task-")

    result_event = None
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Stream events as they arrive: tool calls and text snippets go to the
    # step log so a long run is observable; the `result` event is the
    # final answer.
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(line)
            continue
        etype = event.get("type")
        if etype == "result":
            result_event = event
        elif etype == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if block.get("type") == "text" and block.get("text", "").strip():
                    print(f"[agent] {block['text'].strip()[:400]}")
                elif block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    desc = str(block.get("input", {}).get("description", ""))[:120]
                    print(f"[tool] {name} {desc}".rstrip())

    proc.wait()
    stderr_tail = proc.stderr.read()[-2000:]

    if result_event is None:
        print("No result event from claude CLI.", file=sys.stderr)
        if stderr_tail:
            print(stderr_tail, file=sys.stderr)
        sys.exit(1)

    report = result_event.get("result") or ""
    is_error = bool(result_event.get("is_error")) or (
        result_event.get("subtype") not in (None, "success")
    )

    print(f"::report::{json.dumps(report)}")
    print(f"::success::{json.dumps(not is_error)}")

    if is_error:
        # Surface the failure to the orchestrator so downstream steps skip
        # and the run shows FAILED rather than silently succeeding.
        print(f"Agent finished with an error result: {report[:500]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
