#!/usr/bin/env python3
"""Единый кроссплатформенный запуск RAG-стека.

Скрипт запускает локальный ``llama-server`` и веб-интерфейс Gradio,
дожидается готовности обоих процессов и останавливает их по Ctrl+C.
Работает на Windows и macOS без внешних API и телеметрии.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "T-lite-it-2.1-Q4_K_M.gguf"
DEFAULT_MODEL = BASE_DIR / "Models" / DEFAULT_MODEL_NAME

DEFAULT_LLM_HOST = "127.0.0.1"
DEFAULT_LLM_PORT = 8080
DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 7860
DEFAULT_CTX_SIZE = 8192
DEFAULT_TIMEOUT_S = 300.0

HEALTH_PATH = "/health"
REQUIRED_MODULES = ("gradio", "requests")
LOCAL_HOSTS = {"127.0.0.1", "localhost"}

WINDOWS_CTRL_C_EXIT = 0xC000013A


class LaunchError(RuntimeError):
    """Понятная пользователю ошибка запуска."""


def log(message: str = "") -> None:
    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError):
        # Консоль могла закрыться вместе с терминалом — не мешаем cleanup.
        pass


def default_ngl() -> int:
    """Сколько слоёв отдать GPU.

    На Intel Mac официальные сборки llama.cpp обычно CPU-only, поэтому
    по умолчанию не просим GPU-оффлоад. На Apple Silicon, Windows и Linux
    используется Metal/Vulkan/CUDA.
    """
    is_intel_mac = sys.platform == "darwin" and platform.machine().lower() in {
        "x86_64",
        "i386",
    }
    return 0 if is_intel_mac else 99


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("ожидается положительное целое число")
    return number


def _port(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("порт должен быть в диапазоне 1–65535")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("ожидается положительное число секунд")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Запуск llama-server и веб-интерфейса RAG одной командой.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--llama-server",
        default=os.getenv("LLAMA_SERVER_BIN", ""),
        help="путь к llama-server (по умолчанию ищется в PATH и стандартных местах)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ONPZ_MODEL_PATH", ""),
        help="путь к GGUF-модели (по умолчанию Models/T-lite-it-2.1-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--llama-host",
        default=DEFAULT_LLM_HOST,
        help="локальный хост llama-server",
    )
    parser.add_argument(
        "--llama-port",
        type=_port,
        default=DEFAULT_LLM_PORT,
        help="порт llama-server",
    )
    parser.add_argument(
        "--ui-host",
        default=DEFAULT_UI_HOST,
        help="локальный хост веб-интерфейса",
    )
    parser.add_argument(
        "--ui-port",
        type=_port,
        default=DEFAULT_UI_PORT,
        help="порт веб-интерфейса",
    )
    parser.add_argument(
        "--ctx-size",
        type=_positive_int,
        default=DEFAULT_CTX_SIZE,
        help="размер контекста llama-server",
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=default_ngl(),
        help="число слоёв модели на GPU (0 — только CPU)",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="не передавать llama-server флаг -fa on",
    )
    parser.add_argument(
        "--no-llama",
        action="store_true",
        help="не запускать llama-server, использовать уже запущенный",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="не открывать браузер автоматически",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="создать публичную ссылку Gradio для удалённой проверки",
    )
    parser.add_argument(
        "--share-auth",
        metavar="USER:PASSWORD",
        default="",
        help="логин и пароль для публичной ссылки (только с --share)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT_S,
        help="сколько секунд ждать запуска LLM и интерфейса",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="проверить окружение и выйти без запуска процессов",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    for option in ("llama_host", "ui_host"):
        value = getattr(args, option)
        if value not in LOCAL_HOSTS:
            parser.error(
                f"--{option.replace('_', '-')}: допустим только локальный хост "
                f"(127.0.0.1 или localhost), получено {value!r}"
            )
        setattr(args, option, "127.0.0.1")

    if args.ngl < 0:
        parser.error("--ngl: число слоёв не может быть отрицательным")

    if args.share_auth:
        if not args.share:
            parser.error("--share-auth используется только вместе с --share")
        username, separator, password = args.share_auth.partition(":")
        if not separator or not username or not password:
            parser.error("--share-auth: ожидается формат USER:PASSWORD")

    return args


def missing_modules() -> list[str]:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    return missing


def _is_executable(path: Path) -> bool:
    if not path.is_file():
        return False
    if os.name == "nt":
        return True
    return os.access(path, os.X_OK)


def resolve_model(explicit: str, root: Path = BASE_DIR) -> Path:
    """Найти GGUF-модель и вернуть абсолютный путь."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise LaunchError(
                f"Файл модели не найден: {path}\n"
                "Укажите существующий путь через --model или ONPZ_MODEL_PATH."
            )
        return path

    models_dir = root / "Models"
    default_model = models_dir / DEFAULT_MODEL_NAME
    if default_model.is_file() and default_model.stat().st_size > 0:
        return default_model.resolve()

    candidates = sorted(models_dir.glob("*.gguf")) if models_dir.is_dir() else []
    candidates = [path for path in candidates if path.stat().st_size > 0]
    if len(candidates) == 1:
        return candidates[0].resolve()
    if candidates:
        listing = "\n".join(f"  - {path.name}" for path in candidates)
        raise LaunchError(
            "В Models/ найдено несколько GGUF-моделей, выберите одну:\n"
            f"{listing}\n"
            "Укажите её через --model или ONPZ_MODEL_PATH."
        )

    raise LaunchError(
        f"Модель не найдена: {DEFAULT_MODEL}\n"
        "Скачайте GGUF-модель в Models/ или укажите путь через --model."
    )


def candidate_server_paths(
    root: Path = BASE_DIR, windows: bool | None = None
) -> list[Path]:
    """Стандартные места, где может лежать llama-server."""
    if windows is None:
        windows = os.name == "nt"
    name = "llama-server.exe" if windows else "llama-server"

    paths = [
        root / "bin" / name,
        root / "tools" / "llama.cpp" / name,
        root / "llama.cpp" / name,
        root / "llama.cpp" / "build" / "bin" / name,
        root / "llama.cpp" / "build" / "bin" / "Release" / name,
    ]

    home = Path.home()
    if windows:
        local_appdata = os.getenv("LOCALAPPDATA")
        if local_appdata:
            paths.extend(
                [
                    Path(local_appdata) / "Programs" / "llama.cpp" / name,
                    Path(local_appdata) / "llama.cpp" / name,
                ]
            )
        paths.append(home / "llama.cpp" / "build" / "bin" / name)
    else:
        paths.extend(
            [
                Path("/opt/homebrew/bin") / name,
                Path("/usr/local/bin") / name,
                home / ".local" / "bin" / name,
                home / "llama.cpp" / "build" / "bin" / name,
            ]
        )

    return paths


def _server_help(windows: bool) -> str:
    install_hint = (
        "  Windows: скачайте сборку llama.cpp (Vulkan/CUDA/CPU) со страницы\n"
        "           https://github.com/ggml-org/llama.cpp/releases\n"
        "           и укажите путь к llama-server.exe.\n"
        "  macOS:   brew install llama.cpp"
        if windows
        else "  macOS:   brew install llama.cpp\n"
        "  Linux:   соберите llama.cpp или установите пакет с llama-server."
    )
    env_hint = (
        "  PowerShell: $env:LLAMA_SERVER_BIN = 'C:\\path\\llama-server.exe'"
        if windows
        else "  export LLAMA_SERVER_BIN=/path/to/llama-server"
    )
    return (
        "llama-server не найден.\n"
        f"{install_hint}\n"
        "Также можно добавить llama-server в PATH или задать LLAMA_SERVER_BIN:\n"
        f"{env_hint}\n"
        "Либо передать путь напрямую: --llama-server /path/to/llama-server"
    )


def discover_llama_server(
    explicit: str, root: Path = BASE_DIR, windows: bool | None = None
) -> Path:
    """Найти исполняемый файл llama-server."""
    if windows is None:
        windows = os.name == "nt"

    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise LaunchError(f"llama-server не найден по пути: {path}")
        if not _is_executable(path):
            raise LaunchError(
                f"Файл llama-server не является исполняемым: {path}\n"
                "На macOS/Linux выполните: chmod +x <путь к llama-server>"
            )
        return path

    found = shutil.which("llama-server")
    if found:
        path = Path(found).resolve()
        if _is_executable(path):
            return path

    for candidate in candidate_server_paths(root, windows):
        if _is_executable(candidate):
            return candidate.resolve()

    raise LaunchError(_server_help(windows))


def build_llama_command(
    server: Path,
    model: Path,
    *,
    host: str,
    port: int,
    ctx_size: int = DEFAULT_CTX_SIZE,
    ngl: int = 99,
    flash_attn: bool = True,
) -> list[str]:
    command = [
        str(server),
        "-m",
        str(model),
        "-c",
        str(ctx_size),
        "-ngl",
        str(ngl),
    ]
    if flash_attn:
        command.extend(["-fa", "on"])
    command.extend(["--jinja", "--host", host, "--port", str(port)])
    return command


def build_ui_command(
    python: str,
    root: Path,
    *,
    host: str,
    port: int,
    share: bool = False,
    share_auth: str = "",
) -> list[str]:
    command = [
        python,
        str(root / "App" / "app.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--no-browser",
    ]
    if share:
        command.append("--share")
        if share_auth:
            command.extend(["--auth", share_auth])
    return command


def build_ui_env(
    args: argparse.Namespace, base_env: dict[str, str] | None = None
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["LLM_BASE_URL"] = f"http://{args.llama_host}:{args.llama_port}/v1"
    env["GRADIO_ANALYTICS_ENABLED"] = "False"
    return env


def probe_health(host: str, port: int, timeout: float = 2.0) -> bool:
    url = f"http://{host}:{port}{HEALTH_PATH}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except Exception:
        # Сервер ещё загружается, порт закрыт или ответ не HTTP — это не ошибка.
        return False


def probe_http(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 400
    except Exception:
        return False


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def wait_for(
    check,
    *,
    timeout: float,
    description: str,
    hint: str = "",
    proc: subprocess.Popen | None = None,
    interval: float = 1.0,
    clock=time.monotonic,
    sleep=time.sleep,
) -> None:
    """Ждать, пока check() вернёт True, или понятно упасть по таймауту."""
    started = clock()
    deadline = started + timeout
    next_report = started + 10.0

    while clock() < deadline:
        if proc is not None:
            code = proc.poll()
            if code is not None:
                raise LaunchError(
                    f"{description}: процесс завершился с кодом {code}.\n"
                    f"{hint}".strip()
                )
        if check():
            return

        now = clock()
        if now >= next_report:
            log(f"  … ждём {description.lower()} ({now - started:.0f} с)")
            next_report = now + 10.0
        sleep(interval)

    message = f"{description}: не дождались за {timeout:.0f} с."
    if hint:
        message = f"{message}\n{hint}"
    raise LaunchError(message)


def _popen_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def spawn(
    command: list[str],
    *,
    name: str,
    env: dict[str, str] | None = None,
    cwd: Path = BASE_DIR,
) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            **_popen_kwargs(),
        )
    except OSError as exc:
        raise LaunchError(f"Не удалось запустить {name}: {exc}") from exc


def stop_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    log(f"Останавливаю {name}…")
    try:
        proc.terminate()
    except OSError:
        return
    except KeyboardInterrupt:
        pass
    try:
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
            pass


def _normalize_exit_code(code: int | None) -> int:
    if code is None:
        return 0
    if code < 0 or code in (130, WINDOWS_CTRL_C_EXIT):
        return 130
    return code


def _llm_url(args: argparse.Namespace) -> str:
    return f"http://{args.llama_host}:{args.llama_port}"


def _ui_url(args: argparse.Namespace) -> str:
    return f"http://{args.ui_host}:{args.ui_port}"


def run_check(args: argparse.Namespace) -> int:
    problems: list[str] = []
    llm_problems: list[str] = []
    health_ok = probe_health(args.llama_host, args.llama_port)

    log(f"Проверка окружения · Python {sys.version.split()[0]} · {sys.platform}")

    missing = missing_modules()
    if missing:
        log(f"[FAIL] Python-зависимости: нет {', '.join(missing)}")
        problems.append(
            f"установите зависимости: python -m pip install -r requirements.txt "
            f"(нет: {', '.join(missing)})"
        )
    else:
        log("[OK]   Python-зависимости: gradio, requests")

    if args.no_llama:
        log("[SKIP] llama-server: указан --no-llama")
        log("[SKIP] модель: указан --no-llama")
    else:
        try:
            server = discover_llama_server(args.llama_server)
            log(f"[OK]   llama-server: {server}")
        except LaunchError as exc:
            llm_problems.append(str(exc))
            log(f"[FAIL] llama-server: {str(exc).splitlines()[0]}")

        try:
            model = resolve_model(args.model)
            size_gib = model.stat().st_size / (1024**3)
            log(f"[OK]   модель: {model} ({size_gib:.1f} ГиБ)")
        except (LaunchError, OSError) as exc:
            llm_problems.append(str(exc))
            log(f"[FAIL] модель: {str(exc).splitlines()[0]}")

    if health_ok:
        log(f"[OK]   LLM уже отвечает: {_llm_url(args)}{HEALTH_PATH}")
        if llm_problems:
            log(
                "[WARN] llama-server/модель не найдены, но сейчас будет "
                "использован уже запущенный сервер."
            )
            llm_problems = []
    elif args.no_llama:
        log(
            f"[WARN] LLM не отвечает на {_llm_url(args)}; "
            "интерфейс запустится, но ответы будут с ошибкой."
        )
    elif port_is_free(args.llama_host, args.llama_port):
        log(f"[OK]   порт LLM {args.llama_port} свободен")
    else:
        log(
            f"[FAIL] порт LLM {args.llama_port} занят, "
            "но /health не отвечает"
        )
        llm_problems.append(
            f"порт {args.llama_port} занят другим процессом; "
            "остановите его или укажите --llama-port"
        )

    problems.extend(llm_problems)

    if port_is_free(args.ui_host, args.ui_port):
        log(f"[OK]   порт интерфейса {args.ui_port} свободен")
    else:
        log(f"[FAIL] порт интерфейса {args.ui_port} занят")
        problems.append(
            f"порт {args.ui_port} занят; остановите процесс или укажите --ui-port"
        )

    if problems:
        log(f"\nНайдено проблем: {len(problems)}")
        for problem in problems:
            log(f"  - {problem.splitlines()[0]}")
        return 1

    log("\nВсё готово. Запускайте: python run_app.py")
    return 0


def run(args: argparse.Namespace) -> int:
    if args.check:
        return run_check(args)

    missing = missing_modules()
    if missing:
        raise LaunchError(
            "Нет Python-зависимостей: "
            f"{', '.join(missing)}.\n"
            "Установите их: python -m pip install -r requirements.txt"
        )

    llm_proc: subprocess.Popen | None = None
    ui_proc: subprocess.Popen | None = None
    llm_running = probe_health(args.llama_host, args.llama_port)

    try:
        if args.no_llama:
            if llm_running:
                log(f"Использую запущенный llama-server: {_llm_url(args)}")
            else:
                log(
                    f"⚠ llama-server не отвечает на {_llm_url(args)}. "
                    "Интерфейс запустится, но ответы будут с ошибкой."
                )
        elif llm_running:
            log(
                f"llama-server уже запущен на {_llm_url(args)} — "
                "использую его, второй процесс не запускаю."
            )
        else:
            server = discover_llama_server(args.llama_server)
            model = resolve_model(args.model)

            if not port_is_free(args.llama_host, args.llama_port):
                raise LaunchError(
                    f"Порт {args.llama_port} занят, но llama-server не отвечает "
                    "на /health.\n"
                    "Остановите занявший порт процесс или укажите --llama-port."
                )

            command = build_llama_command(
                server,
                model,
                host=args.llama_host,
                port=args.llama_port,
                ctx_size=args.ctx_size,
                ngl=args.ngl,
                flash_attn=not args.no_flash_attn,
            )
            log("Запускаю llama-server:")
            log("  " + " ".join(command))
            llm_proc = spawn(command, name="llama-server")
            wait_for(
                lambda: probe_health(args.llama_host, args.llama_port),
                timeout=args.timeout,
                description="загрузка llama-server",
                hint=(
                    "Проверьте сообщения llama-server выше. Если модель большая, "
                    "увеличьте --timeout."
                ),
                proc=llm_proc,
            )
            log(f"llama-server готов: {_llm_url(args)}")

        if not port_is_free(args.ui_host, args.ui_port):
            raise LaunchError(
                f"Порт интерфейса {args.ui_port} уже занят.\n"
                "Остановите старый процесс или укажите другой порт: --ui-port 7861"
            )

        ui_command = build_ui_command(
            sys.executable,
            BASE_DIR,
            host=args.ui_host,
            port=args.ui_port,
            share=args.share,
            share_auth=args.share_auth,
        )
        log("Запускаю веб-интерфейс:")
        log("  " + " ".join(ui_command))
        ui_proc = spawn(
            ui_command,
            name="веб-интерфейс",
            env=build_ui_env(args),
        )

        wait_for(
            lambda: probe_http(_ui_url(args) + "/"),
            timeout=args.timeout,
            description="запуск веб-интерфейса",
            hint="Проверьте сообщения Gradio выше.",
            proc=ui_proc,
        )

        ui_url = _ui_url(args)
        log(f"Интерфейс доступен локально: {ui_url}")
        if args.share:
            log(
                "Публичная ссылка для куратора появится ниже в логе — в строке "
                "«Публичная ссылка для проверки»."
            )
            log("Она работает до недели и только пока этот компьютер включён.")
        if not args.no_browser:
            if not webbrowser.open(ui_url):
                log("Не удалось открыть браузер автоматически — откройте URL вручную.")
        log("Нажмите Ctrl+C, чтобы остановить интерфейс и llama-server.")

        return _normalize_exit_code(ui_proc.wait())
    finally:
        stop_process(ui_proc, "веб-интерфейс")
        stop_process(llm_proc, "llama-server")


def _install_signal_handlers() -> None:
    """Превратить SIGTERM/SIGBREAK в KeyboardInterrupt, чтобы сработал finally."""

    def stop(_signum, _frame) -> None:
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, stop)
        except (OSError, ValueError):
            pass


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    _install_signal_handlers()
    args = parse_args(argv)
    try:
        return run(args)
    except LaunchError as exc:
        print(f"\nОшибка: {exc}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        log("\nОстановлено пользователем.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
