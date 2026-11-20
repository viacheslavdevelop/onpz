"""Тесты лаунчера без запуска реальных процессов и сетевых вызовов."""

import contextlib
import io
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Tests import _bootstrap  # noqa: F401

import run_app


def make_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o755)
    return path


class ParseArgsTest(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            args = run_app.parse_args([])
        self.assertEqual(args.llama_host, "127.0.0.1")
        self.assertEqual(args.llama_port, 8080)
        self.assertEqual(args.ui_host, "127.0.0.1")
        self.assertEqual(args.ui_port, 7860)
        self.assertEqual(args.ctx_size, 8192)
        self.assertFalse(args.no_llama)
        self.assertFalse(args.check)

    def test_localhost_is_normalized(self) -> None:
        args = run_app.parse_args(["--ui-host", "localhost", "--llama-host", "localhost"])
        self.assertEqual(args.ui_host, "127.0.0.1")
        self.assertEqual(args.llama_host, "127.0.0.1")

    def test_remote_host_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_app.parse_args(["--ui-host", "0.0.0.0"])

    def test_invalid_port_is_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_app.parse_args(["--ui-port", "70000"])

    def test_share_is_disabled_by_default(self) -> None:
        args = run_app.parse_args([])
        self.assertFalse(args.share)
        self.assertEqual(args.share_auth, "")

    def test_share_auth_requires_share(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_app.parse_args(["--share-auth", "curator:secret"])

    def test_share_auth_requires_colon(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_app.parse_args(["--share", "--share-auth", "curator"])

    def test_share_auth_is_accepted_with_share(self) -> None:
        args = run_app.parse_args(["--share", "--share-auth", "curator:secret"])
        self.assertTrue(args.share)
        self.assertEqual(args.share_auth, "curator:secret")

    def test_environment_defaults(self) -> None:
        env = {
            "LLAMA_SERVER_BIN": "/custom/llama-server",
            "ONPZ_MODEL_PATH": "/custom/model.gguf",
        }
        with patch.dict(os.environ, env, clear=True):
            args = run_app.parse_args([])
        self.assertEqual(args.llama_server, "/custom/llama-server")
        self.assertEqual(args.model, "/custom/model.gguf")


class ResolveModelTest(unittest.TestCase):
    def test_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "Models" / run_app.DEFAULT_MODEL_NAME
            model.parent.mkdir(parents=True)
            model.write_bytes(b"gguf")
            self.assertEqual(run_app.resolve_model("", root), model.resolve())

    def test_single_gguf_is_used_as_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "Models" / "other.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"gguf")
            self.assertEqual(run_app.resolve_model("", root), model.resolve())

    def test_several_gguf_require_explicit_choice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "Models"
            models.mkdir(parents=True)
            (models / "a.gguf").write_bytes(b"a")
            (models / "b.gguf").write_bytes(b"b")
            with self.assertRaises(run_app.LaunchError) as ctx:
                run_app.resolve_model("", root)
            self.assertIn("несколько", str(ctx.exception))

    def test_explicit_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "Models" / "custom.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"gguf")
            self.assertEqual(
                run_app.resolve_model("Models/custom.gguf", root), model.resolve()
            )

    def test_missing_model_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(run_app.LaunchError):
                run_app.resolve_model("", Path(tmp))


class DiscoverServerTest(unittest.TestCase):
    def test_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = make_executable(root / "bin" / "llama-server")
            self.assertEqual(
                run_app.discover_llama_server(str(server), root), server.resolve()
            )

    def test_missing_explicit_path_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(run_app.LaunchError):
                run_app.discover_llama_server(
                    str(Path(tmp) / "missing" / "llama-server"), Path(tmp)
                )

    def test_path_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = make_executable(root / "llama-server")
            with patch("run_app.shutil.which", return_value=str(server)):
                self.assertEqual(run_app.discover_llama_server("", root), server.resolve())

    def test_project_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "llama-server.exe" if os.name == "nt" else "llama-server"
            server = make_executable(root / "tools" / "llama.cpp" / name)
            with patch("run_app.shutil.which", return_value=None):
                self.assertEqual(
                    run_app.discover_llama_server("", root), server.resolve()
                )

    def test_not_found_has_install_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("run_app.shutil.which", return_value=None):
                with self.assertRaises(run_app.LaunchError) as ctx:
                    run_app.discover_llama_server("", Path(tmp))
        self.assertIn("llama-server не найден", str(ctx.exception))
        self.assertIn("LLAMA_SERVER_BIN", str(ctx.exception))


class BuildCommandsTest(unittest.TestCase):
    def test_llama_command_matches_documented_flags(self) -> None:
        command = run_app.build_llama_command(
            Path("/x/llama-server"),
            Path("/x/model.gguf"),
            host="127.0.0.1",
            port=8080,
            ctx_size=8192,
            ngl=99,
            flash_attn=True,
        )
        self.assertEqual(
            command,
            [
                str(Path("/x/llama-server")),
                "-m",
                str(Path("/x/model.gguf")),
                "-c",
                "8192",
                "-ngl",
                "99",
                "-fa",
                "on",
                "--jinja",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
            ],
        )

    def test_flash_attention_can_be_disabled(self) -> None:
        command = run_app.build_llama_command(
            Path("/x/llama-server"),
            Path("/x/model.gguf"),
            host="127.0.0.1",
            port=8080,
            flash_attn=False,
        )
        self.assertNotIn("-fa", command)

    def test_ui_command_always_disables_gradio_browser(self) -> None:
        command = run_app.build_ui_command(
            "python", Path("/repo"), host="127.0.0.1", port=7861
        )
        self.assertEqual(
            command,
            [
                "python",
                str(Path("/repo") / "App" / "app.py"),
                "--host",
                "127.0.0.1",
                "--port",
                "7861",
                "--no-browser",
            ],
        )

    def test_ui_command_passes_share_flags(self) -> None:
        command = run_app.build_ui_command(
            "python",
            Path("/repo"),
            host="127.0.0.1",
            port=7860,
            share=True,
            share_auth="curator:secret",
        )
        self.assertEqual(
            command,
            [
                "python",
                str(Path("/repo") / "App" / "app.py"),
                "--host",
                "127.0.0.1",
                "--port",
                "7860",
                "--no-browser",
                "--share",
                "--auth",
                "curator:secret",
            ],
        )

    def test_ui_env_points_to_custom_llm_port(self) -> None:
        args = run_app.parse_args(["--llama-port", "9090"])
        env = run_app.build_ui_env(args, base_env={"PATH": "keep-me"})
        self.assertEqual(env["LLM_BASE_URL"], "http://127.0.0.1:9090/v1")
        self.assertEqual(env["GRADIO_ANALYTICS_ENABLED"], "False")
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PATH"], "keep-me")


class WaitForTest(unittest.TestCase):
    def test_returns_when_check_succeeds(self) -> None:
        run_app.wait_for(
            lambda: True,
            timeout=1,
            description="тест",
            clock=lambda: 0.0,
            sleep=lambda _seconds: None,
        )

    def test_timeout_raises_with_hint(self) -> None:
        now = [0.0]

        def clock() -> float:
            return now[0]

        def sleep(seconds: float) -> None:
            now[0] += seconds

        with self.assertRaises(run_app.LaunchError) as ctx:
            run_app.wait_for(
                lambda: False,
                timeout=2,
                description="запуск теста",
                hint="подсказка",
                interval=1.0,
                clock=clock,
                sleep=sleep,
            )
        self.assertIn("не дождались", str(ctx.exception))
        self.assertIn("подсказка", str(ctx.exception))

    def test_exited_process_is_reported_immediately(self) -> None:
        class ExitedProcess:
            def poll(self):
                return 3

        with self.assertRaises(run_app.LaunchError) as ctx:
            run_app.wait_for(
                lambda: False,
                timeout=30,
                description="запуск теста",
                proc=ExitedProcess(),
                clock=lambda: 0.0,
                sleep=lambda _seconds: None,
            )
        self.assertIn("кодом 3", str(ctx.exception))


class PortTest(unittest.TestCase):
    def test_busy_port_is_detected(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            self.assertFalse(run_app.port_is_free("127.0.0.1", port))


if __name__ == "__main__":
    unittest.main()
