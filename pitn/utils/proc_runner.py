# -*- coding: utf-8 -*-
import atexit
import datetime
import os
import queue
import shlex
import signal
import subprocess
import threading
from pathlib import Path

from box import Box

import docker

DOCKER_CLIENT = None

docker_run_default_config = dict(
    # detach=True,
    ipc_mode="host",
    pid_mode="host",
    remove=True,
    auto_remove=True,
    security_opt=["seccomp=unconfined"],
    cap_add=["SYS_PTRACE"],
    volumes={"/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"}},
    # tty=True,
    # stderr=True,
    privileged=True,
    # stdout=True,
)


def call_docker_run(
    img: str,
    cmd: str,
    env: dict = dict(),
    run_config: dict = dict(),
):
    global DOCKER_CLIENT
    if DOCKER_CLIENT is None:
        DOCKER_CLIENT = docker.from_env()

    client = DOCKER_CLIENT

    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    # Merge user run config and the default config. Merginb Box objects ensures
    # sub-dicts are merged properly.
    default_config = Box(docker_run_default_config)
    user_config = Box(run_config)
    run_opts = default_config + user_config
    # Some options are lists, and lists should be appended.
    for k in {"security_opts", "cap_add", "mounts"}:
        if k in run_opts.keys():
            run_opts[k] = default_config.get(k, list()) + user_config.get(k, list())
    run_opts["environment"] = env
    run_opts = run_opts.to_dict()

    # Run container in detached mode.
    container = client.containers.run(img, cmd, detach=True, **run_opts)

    tail_logs = list()
    tail_maxsize = 30
    for line in container.logs(stream=True, stdout=True, stderr=True):
        print(
            f"Docker {datetime.datetime.now().replace(microsecond=0)} | ",
            line.decode().strip(),
        )
        tail_logs.append(line.decode().strip())
        if len(tail_logs) > tail_maxsize:
            tail_logs.pop(0)
    exit_status = container.wait()["StatusCode"]
    if exit_status > 0:
        raise docker.errors.ContainerError(
            container,
            exit_status,
            cmd,
            img,
            "\n".join(tail_logs),
        )

    return 0


def get_docker_mount_obj(path: Path, **mount_obj_kwargs):
    p = Path(path).resolve()
    m_obj = None
    if p.is_file():
        kw_defaults = dict(target=str(p), type="bind")
        mount_kw = {**kw_defaults, **mount_obj_kwargs}
        m_obj = docker.types.Mount(source=str(p), **mount_kw)
    elif p.is_dir():
        kw_defaults = dict(mode="rw")
        vol_kw = {**kw_defaults, **mount_obj_kwargs}
        m_obj = {"bind": str(p), **vol_kw}
    else:
        raise ValueError(f"ERROR: {path} is invalid path")

    return m_obj


def call_shell_exec(
    cmd: str,
    args: str,
    cwd: Path,
    env: dict = None,
    prefix: str = "",
    popen_args_override=None,
):
    env = os.environ if env is None else env

    # Taken from
    # <https://sharats.me/posts/the-ever-useful-and-neat-subprocess-module/#watching-both-stdout-and-stderr>
    io_queue = queue.Queue()

    def stream_watcher(identifier, stream):
        for line in stream:
            io_queue.put((identifier, line))
        if not stream.closed:
            stream.close()

    # Split the prefix up into tokens, with the cmd + args as the "string to run"
    # argument to the prefix.
    # Start the new process and assign it to a new process group. This allows for
    # killing child processes; see <https://stackoverflow.com/a/34027738/13225248>
    if popen_args_override is None:
        popen_args = shlex.split(prefix) + [cmd + " " + args]
    else:
        popen_args = popen_args_override
    proc = subprocess.Popen(
        popen_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        bufsize=10,
        text=True,
        preexec_fn=os.setpgrp,
    )

    out_lines = list()

    def printer():
        while True:
            try:
                # Block for a few seconds.
                item = io_queue.get(True, 2)
            except queue.Empty:
                # No output in either streams for a few seconds. Are we done?
                if proc.poll() is not None:
                    break
            else:
                identifier, line = item
                log_line = f"{identifier}: {line}"
                print(log_line.rstrip(), flush=True)
                out_lines.append(log_line)

    t_stdout = threading.Thread(
        target=stream_watcher, name="stdout-watcher", args=("STDOUT", proc.stdout)
    )
    t_stdout.start()

    t_stderr = threading.Thread(
        target=stream_watcher, name="stderr-watcher", args=("STDERR", proc.stderr)
    )
    t_stderr.start()

    t_print = threading.Thread(target=printer, name="printer")
    t_print.start()

    # Make sure proc is killed when exiting.
    @atexit.register
    def kill_proc():
        print("Maybe kill proc? ", proc.pid, flush=True)
        if proc.poll() is None:
            p_group_id = os.getpgid(proc.pid)
            os.killpg(p_group_id, signal.SIGKILL)
            print(
                "Killed proc ", proc.pid, " and process group ", p_group_id, flush=True
            )
            # proc.terminate()
            # proc.kill()

    t_stdout.join()
    t_stderr.join()
    t_print.join()

    return_code = proc.poll()
    # Proc should be done by now, if threads have exited.
    if return_code is None:
        kill_proc()

    # Proc absolutely has to be dead by now.
    atexit.unregister(kill_proc)
    if return_code > 0:
        raise subprocess.CalledProcessError(
            return_code, proc.args, "\n".join(out_lines)
        )

    return out_lines
