from __future__ import annotations

# ruff: noqa: E402, I001

import json
import os
import queue
import signal
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import gi

gi.require_version("GLib", "2.0")
gi.require_version("GObject", "2.0")
from gi.repository import GLib, GObject

from visionmodelquest.explorer.lifecycle import WorkerState, automatic_restart_delay
from visionmodelquest.explorer.logging import local_logger
from visionmodelquest.explorer.protocol import PROTOCOL_VERSION


class WorkerController(GObject.Object):
    __gsignals__ = {
        "state-changed": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "response": (GObject.SignalFlags.RUN_FIRST, None, (object,)),
        "output-fragment": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "worker-error": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
    }

    def __init__(
        self,
        *,
        python: Path,
        model_key: str,
        session_root: Path,
        runtime_root: Path,
        log_root: Path,
        cache_root: Path | None = None,
        idle_seconds: int = 600,
    ) -> None:
        super().__init__()
        # Preserve the virtual-environment launcher. Resolving its interpreter symlink
        # would execute the base pyenv Python and discard the venv's site-packages.
        self.python = python.absolute()
        self.model_key = model_key
        self.session_root = session_root.resolve()
        self.runtime_root = runtime_root.resolve()
        self.log_root = log_root.resolve()
        self.cache_root = cache_root.resolve() if cache_root else None
        self.idle_seconds = idle_seconds
        self.process: subprocess.Popen[bytes] | None = None
        self.state = WorkerState.STOPPED
        self._stdout = b""
        self._stderr = b""
        self._pending: dict[str, str] = {}
        self._auto_unload_ids: set[str] = set()
        self._write_queue: queue.Queue[bytes | None] = queue.Queue()
        self._restart_after_exit = False
        self._shutting_down = False
        self._consecutive_crashes = 0
        self._startup_stderr: list[str] = []
        self._idle_source: int | None = None
        self.log = local_logger("application", self.log_root)
        self.protocol_log = local_logger("protocol", self.log_root)

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if not self.python.is_file():
            self.emit("worker-error", f"Missing inference environment: {self.python}")
            return
        self._shutting_down = False
        self._set_state(WorkerState.STARTING)
        command = [
            str(self.python),
            "-m",
            "visionmodelquest.experiment_worker",
            "--model-key",
            self.model_key,
            "--session-root",
            str(self.session_root),
            "--log-root",
            str(self.log_root),
        ]
        if self.cache_root:
            command.extend(["--cache-root", str(self.cache_root)])
        environment = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                start_new_session=True,
            )
        except OSError as error:
            self._set_state(WorkerState.FAILED)
            self.emit("worker-error", f"Could not start the inference worker: {error}")
            return
        self._write_pid()
        assert self.process.stdout and self.process.stderr
        os.set_blocking(self.process.stdout.fileno(), False)
        os.set_blocking(self.process.stderr.fileno(), False)
        GLib.io_add_watch(
            self.process.stdout.fileno(),
            GLib.IOCondition.IN | GLib.IOCondition.HUP | GLib.IOCondition.ERR,
            self._read_stdout,
        )
        GLib.io_add_watch(
            self.process.stderr.fileno(),
            GLib.IOCondition.IN | GLib.IOCondition.HUP | GLib.IOCondition.ERR,
            self._read_stderr,
        )
        GLib.child_watch_add(self.process.pid, self._child_exited)
        threading.Thread(target=self._writer, daemon=True).start()
        self.request("initialise_processor")

    def request(self, operation: str, payload: dict[str, Any] | None = None) -> str:
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("inference worker is not running")
        request_id = uuid.uuid4().hex
        envelope = {
            "protocol_version": PROTOCOL_VERSION,
            "kind": "request",
            "request_id": request_id,
            "operation": operation,
            "model_key": self.model_key,
            "payload": payload or {},
        }
        encoded = (json.dumps(envelope, separators=(",", ":")) + "\n").encode()
        if len(encoded) > 64 * 1024:
            raise ValueError("request exceeds the protocol limit")
        self._pending[request_id] = operation
        self.protocol_log.info("request id=%s operation=%s", request_id, operation)
        self._write_queue.put(encoded)
        if operation == "load_model":
            self._set_state(WorkerState.LOADING_MODEL)
        elif operation == "generate":
            self._set_state(WorkerState.GENERATING)
            self._arm_idle_timer()
        elif operation == "unload_model":
            self._set_state(WorkerState.UNLOADING)
        return request_id

    def cancel(self) -> None:
        if self.state != WorkerState.GENERATING or not self.process:
            return
        self._set_state(WorkerState.CANCELLING)
        self.log.info("generation_cancelled action=terminate_worker_process_group")
        self._restart_after_exit = True
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def stop(self, *, restart: bool = False) -> None:
        self._restart_after_exit = restart
        if not self.process or self.process.poll() is not None:
            self._set_state(WorkerState.STOPPED)
            if restart:
                GLib.idle_add(self.start)
            return
        if not restart and self.state in {
            WorkerState.PROCESSOR_READY,
            WorkerState.MODEL_READY,
        }:
            try:
                self.request("shutdown")
                GLib.timeout_add_seconds(3, self._force_stop)
                return
            except RuntimeError:
                pass
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def change_model(self, model_key: str) -> None:
        if model_key == self.model_key:
            return
        self.model_key = model_key
        self._restart_after_exit = True
        self.stop(restart=True)

    def close(self) -> None:
        self._shutting_down = True
        self._restart_after_exit = False
        process = self.process
        if not process or process.poll() is not None:
            self._remove_pid()
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=2)
        finally:
            self.process = None
            self._remove_pid()

    def _writer(self) -> None:
        while True:
            value = self._write_queue.get()
            if value is None:
                return
            process = self.process
            if not process or not process.stdin:
                continue
            try:
                process.stdin.write(value)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                GLib.idle_add(self.emit, "worker-error", "The inference worker disconnected.")

    def _read_stdout(self, descriptor: int, condition: GLib.IOCondition) -> bool:
        self._stdout += self._read_available(descriptor)
        while b"\n" in self._stdout:
            raw, self._stdout = self._stdout.split(b"\n", 1)
            if raw:
                self._handle_line(raw)
        return not bool(condition & (GLib.IOCondition.HUP | GLib.IOCondition.ERR))

    def _read_stderr(self, descriptor: int, condition: GLib.IOCondition) -> bool:
        self._stderr += self._read_available(descriptor)
        while b"\n" in self._stderr:
            raw, self._stderr = self._stderr.split(b"\n", 1)
            if raw:
                if self.state in {
                    WorkerState.STARTING,
                    WorkerState.RESTARTING,
                }:
                    text = raw.decode("utf-8", errors="replace")[:1_000]
                    self._startup_stderr.append(text)
                    self._startup_stderr = self._startup_stderr[-8:]
                    self.log.error("worker_startup_stderr %s", text)
                else:
                    self.log.warning("worker_stderr length=%s", len(raw))
        return not bool(condition & (GLib.IOCondition.HUP | GLib.IOCondition.ERR))

    @staticmethod
    def _read_available(descriptor: int) -> bytes:
        try:
            return os.read(descriptor, 65_536)
        except BlockingIOError:
            return b""

    def _handle_line(self, raw: bytes) -> None:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.emit("worker-error", "The worker returned a malformed protocol message.")
            return
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            self.emit("worker-error", "The worker protocol version is incompatible.")
            return
        request_id = payload.get("request_id", "")
        kind = payload.get("kind")
        if kind == "event" and payload.get("event") == "output_fragment":
            text = payload.get("payload", {}).get("text")
            if isinstance(text, str):
                self.emit("output-fragment", text)
            return
        if kind != "response" or request_id not in self._pending:
            self.protocol_log.warning("ignored stale_or_unknown_response id=%s", request_id)
            return
        operation = self._pending.pop(request_id)
        if request_id in self._auto_unload_ids:
            payload["auto_unload"] = True
            self._auto_unload_ids.discard(request_id)
        self.protocol_log.info(
            "response id=%s operation=%s status=%s",
            request_id,
            operation,
            payload.get("status"),
        )
        try:
            self._set_state(WorkerState(payload.get("worker_state", WorkerState.FAILED)))
        except ValueError:
            self._set_state(WorkerState.FAILED)
        self.emit("response", payload)
        if operation == "initialise_processor" and payload.get("status") == "ok":
            self._consecutive_crashes = 0
            self._startup_stderr.clear()
        if (
            operation in {"load_model", "generate"}
            or (operation == "inspect_image" and self.state == WorkerState.MODEL_READY)
        ) and payload.get("status") == "ok":
            self._arm_idle_timer()

    def _child_exited(self, pid: int, status: int) -> None:
        del pid
        expected = self._shutting_down or self._restart_after_exit
        self.log.info("worker_exited status=%s expected=%s", status, expected)
        self.process = None
        self._pending.clear()
        self._remove_pid()
        self._write_queue.put(None)
        if self._restart_after_exit and not self._shutting_down:
            self._restart_after_exit = False
            self._set_state(WorkerState.RESTARTING)
            GLib.timeout_add(250, self._restart)
        elif expected:
            self._set_state(WorkerState.STOPPED)
        else:
            self._set_state(WorkerState.FAILED)
            self._consecutive_crashes += 1
            restart_delay = automatic_restart_delay(self._consecutive_crashes)
            if self._consecutive_crashes == 1:
                self.emit(
                    "worker-error",
                    "The inference worker stopped unexpectedly. A clean restart will be attempted.",
                )
            if restart_delay is not None:
                self._set_state(WorkerState.RESTARTING)
                GLib.timeout_add_seconds(restart_delay, self._restart)
            else:
                self.emit(
                    "worker-error",
                    "The inference worker failed repeatedly. Automatic restart has stopped; "
                    "see the local application log for startup diagnostics.",
                )

    def _restart(self) -> bool:
        self.start()
        return GLib.SOURCE_REMOVE

    def _force_stop(self) -> bool:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return GLib.SOURCE_REMOVE

    def _arm_idle_timer(self) -> None:
        if self._idle_source:
            GLib.source_remove(self._idle_source)
        self._idle_source = GLib.timeout_add_seconds(self.idle_seconds, self._idle_unload)

    def _idle_unload(self) -> bool:
        self._idle_source = None
        if self.state == WorkerState.MODEL_READY:
            self._auto_unload_ids.add(self.request("unload_model"))
        return GLib.SOURCE_REMOVE

    def _set_state(self, state: WorkerState) -> None:
        if self.state != state:
            self.state = state
            self.emit("state-changed", state.value)

    def _write_pid(self) -> None:
        if not self.process:
            return
        self.runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        pid_path = self.runtime_root / "worker.pid"
        pid_path.write_text(
            json.dumps(
                {
                    "pid": self.process.pid,
                    "model_key": self.model_key,
                    "session_root": str(self.session_root),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        pid_path.chmod(0o600)

    def _remove_pid(self) -> None:
        (self.runtime_root / "worker.pid").unlink(missing_ok=True)
