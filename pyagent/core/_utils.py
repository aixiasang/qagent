import base64
import os
import re
import asyncio
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime


class AgentLogger:
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        enabled: bool = True,
        log_file: Optional[str] = None,
        format: Optional[str] = None,
    ):
        self.enabled = enabled
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        self.logger.handlers.clear()

        if not enabled:
            self.logger.addHandler(logging.NullHandler())
            return

        if format is None:
            format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

        formatter = logging.Formatter(format)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg: str, **kwargs):
        if self.enabled:
            self.logger.info(msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        if self.enabled:
            self.logger.debug(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        if self.enabled:
            self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        if self.enabled:
            self.logger.error(msg, **kwargs)

    def disable(self):
        self.enabled = False
        self.logger.handlers.clear()
        self.logger.addHandler(logging.NullHandler())

    def enable(self):
        self.enabled = True


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def video_to_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class FileOperations:
    @staticmethod
    async def read_file(filepath: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Read entire file content.

        Args:
            filepath (str): Path to the file
            encoding (str): File encoding (default: utf-8)

        Returns:
            Dict with success status, content, filepath, and size
        """
        try:

            def _read():
                with open(filepath, "r", encoding=encoding) as f:
                    return f.read()

            content = await asyncio.to_thread(_read)
            return {
                "success": True,
                "content": content,
                "filepath": filepath,
                "size": len(content),
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "FileNotFoundError",
                "message": f"File not found: {filepath}",
            }
        except PermissionError:
            return {
                "success": False,
                "error": "PermissionError",
                "message": f"Permission denied: {filepath}",
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def read_file_lines(
        filepath: str,
        start_line: int = 1,
        end_line: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """
        Read specific lines from file.

        Args:
            filepath (str): Path to the file
            start_line (int): Starting line number (1-indexed)
            end_line (int): Ending line number (optional)
            encoding (str): File encoding

        Returns:
            Dict with success status, lines, and line numbers
        """
        try:

            def _read():
                with open(filepath, "r", encoding=encoding) as f:
                    lines = f.readlines()
                    if end_line is None:
                        return lines[start_line - 1 :]
                    return lines[start_line - 1 : end_line]

            lines = await asyncio.to_thread(_read)
            return {
                "success": True,
                "lines": lines,
                "total_lines": len(lines),
                "start_line": start_line,
                "end_line": end_line or start_line + len(lines) - 1,
            }
        except IndexError:
            return {
                "success": False,
                "error": "IndexError",
                "message": "Line number out of range",
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def write_file(
        filepath: str, content: str, encoding: str = "utf-8", create_dirs: bool = True
    ) -> Dict[str, Any]:
        """
        Write content to file, creating parent directories if needed.

        Args:
            filepath (str): Path to the file
            content (str): Content to write
            encoding (str): File encoding
            create_dirs (bool): Create parent directories if they don't exist

        Returns:
            Dict with success status, filepath, and bytes written
        """
        try:

            def _write():
                path = Path(filepath)
                if create_dirs:
                    path.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w", encoding=encoding) as f:
                    f.write(content)
                return os.path.getsize(filepath)

            size = await asyncio.to_thread(_write)
            return {"success": True, "filepath": filepath, "bytes_written": size}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def append_file(filepath: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """
        Append content to existing file.

        Args:
            filepath (str): Path to the file
            content (str): Content to append
            encoding (str): File encoding

        Returns:
            Dict with success status, filepath, and new file size
        """
        try:

            def _append():
                with open(filepath, "a", encoding=encoding) as f:
                    f.write(content)
                return os.path.getsize(filepath)

            size = await asyncio.to_thread(_append)
            return {"success": True, "filepath": filepath, "new_size": size}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def delete_file(filepath: str) -> Dict[str, Any]:
        """
        Delete a file.

        Args:
            filepath (str): Path to the file to delete

        Returns:
            Dict with success status and deletion confirmation
        """
        try:
            if not os.path.exists(filepath):
                return {
                    "success": False,
                    "error": "FileNotFoundError",
                    "message": f"File not found: {filepath}",
                }

            if os.path.isdir(filepath):
                return {
                    "success": False,
                    "error": "IsADirectoryError",
                    "message": f"Path is a directory: {filepath}",
                }

            await asyncio.to_thread(os.remove, filepath)
            return {"success": True, "filepath": filepath, "status": "deleted"}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def copy_file(src: str, dst: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Copy file from source to destination.

        Args:
            src (str): Source file path
            dst (str): Destination file path
            overwrite (bool): Allow overwriting existing files

        Returns:
            Dict with success status, source, and destination paths
        """
        try:
            if not os.path.exists(src):
                return {
                    "success": False,
                    "error": "FileNotFoundError",
                    "message": f"Source not found: {src}",
                }

            if os.path.exists(dst) and not overwrite:
                return {
                    "success": False,
                    "error": "FileExistsError",
                    "message": f"Destination exists: {dst}",
                }

            await asyncio.to_thread(shutil.copy2, src, dst)
            return {"success": True, "from": src, "to": dst, "status": "copied"}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def move_file(src: str, dst: str, overwrite: bool = False) -> Dict[str, Any]:
        """
        Move file from source to destination.

        Args:
            src (str): Source file path
            dst (str): Destination file path
            overwrite (bool): Allow overwriting existing files

        Returns:
            Dict with success status, source, and destination paths
        """
        try:
            if not os.path.exists(src):
                return {
                    "success": False,
                    "error": "FileNotFoundError",
                    "message": f"Source not found: {src}",
                }

            if os.path.exists(dst) and not overwrite:
                return {
                    "success": False,
                    "error": "FileExistsError",
                    "message": f"Destination exists: {dst}",
                }

            await asyncio.to_thread(shutil.move, src, dst)
            return {"success": True, "from": src, "to": dst, "status": "moved"}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def file_exists(filepath: str) -> Dict[str, Any]:
        """
        Check if file exists.

        Args:
            filepath (str): Path to check

        Returns:
            Dict with success status and exists boolean
        """
        try:
            exists = await asyncio.to_thread(os.path.exists, filepath)
            return {"success": True, "exists": exists, "filepath": filepath}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def get_file_info(filepath: str) -> Dict[str, Any]:
        """
        Get detailed file information.

        Args:
            filepath (str): Path to the file

        Returns:
            Dict with name, path, size, timestamps, type, and extension
        """
        try:

            def _get_info():
                path = Path(filepath)
                stat = path.stat()
                return {
                    "name": path.name,
                    "path": str(path.absolute()),
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "is_file": path.is_file(),
                    "is_dir": path.is_dir(),
                    "extension": path.suffix,
                }

            info = await asyncio.to_thread(_get_info)
            return {"success": True, **info}
        except FileNotFoundError:
            return {
                "success": False,
                "error": "FileNotFoundError",
                "message": f"Path not found: {filepath}",
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def replace_in_file(
        filepath: str,
        old_text: str,
        new_text: str,
        count: int = -1,
        encoding: str = "utf-8",
    ) -> Dict[str, Any]:
        """
        Replace text in file.

        Args:
            filepath (str): Path to the file
            old_text (str): Text to replace
            new_text (str): Replacement text
            count (int): Maximum number of replacements (-1 for all)
            encoding (str): File encoding

        Returns:
            Dict with success status, occurrences found, and replacements made
        """
        try:

            def _replace():
                with open(filepath, "r", encoding=encoding) as f:
                    content = f.read()

                occurrences = content.count(old_text)
                new_content = content.replace(old_text, new_text, count)

                with open(filepath, "w", encoding=encoding) as f:
                    f.write(new_content)

                return occurrences, len(new_content)

            occurrences, new_size = await asyncio.to_thread(_replace)
            return {
                "success": True,
                "filepath": filepath,
                "occurrences": occurrences,
                "replaced": occurrences if count == -1 else min(occurrences, count),
                "new_size": new_size,
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}


class DirectoryOperations:
    """Async directory operations: list, create, delete, search, tree view."""

    @staticmethod
    async def list_directory(
        dirpath: str = ".",
        recursive: bool = False,
        show_hidden: bool = False,
        pattern: str = "*",
    ) -> Dict[str, Any]:
        """
        List files and directories in a path.

        Args:
            dirpath (str): Directory path to list
            recursive (bool): Include subdirectories recursively
            show_hidden (bool): Show hidden files (starting with .)
            pattern (str): Glob pattern for filtering

        Returns:
            Dict with success status, items list, and count
        """
        try:

            def _list():
                path = Path(dirpath)
                if not path.exists():
                    raise FileNotFoundError(f"Directory not found: {dirpath}")

                if recursive:
                    glob_pattern = f"**/{pattern}"
                else:
                    glob_pattern = pattern

                results = []
                for item in path.glob(glob_pattern):
                    if not show_hidden and item.name.startswith("."):
                        continue

                    try:
                        stat = item.stat()
                        results.append(
                            {
                                "name": item.name,
                                "path": str(item.absolute()),
                                "type": "dir" if item.is_dir() else "file",
                                "size": stat.st_size if item.is_file() else None,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            }
                        )
                    except Exception:
                        continue

                return results

            items = await asyncio.to_thread(_list)
            return {
                "success": True,
                "directory": dirpath,
                "items": items,
                "count": len(items),
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def create_directory(
        dirpath: str, parents: bool = True, exist_ok: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new directory.

        Args:
            dirpath (str): Path for the new directory
            parents (bool): Create parent directories if needed
            exist_ok (bool): Don't error if directory already exists

        Returns:
            Dict with success status and path
        """
        try:
            await asyncio.to_thread(Path(dirpath).mkdir, parents=parents, exist_ok=exist_ok)
            return {"success": True, "path": dirpath, "status": "created"}
        except FileExistsError:
            return {
                "success": False,
                "error": "FileExistsError",
                "message": f"Directory exists: {dirpath}",
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def delete_directory(dirpath: str, recursive: bool = False) -> Dict[str, Any]:
        """
        Delete a directory.

        Args:
            dirpath (str): Directory path to delete
            recursive (bool): Delete contents recursively

        Returns:
            Dict with success status and path
        """
        try:

            def _delete():
                path = Path(dirpath)
                if not path.exists():
                    raise FileNotFoundError(f"Directory not found: {dirpath}")

                if not path.is_dir():
                    raise NotADirectoryError(f"Not a directory: {dirpath}")

                if recursive:
                    shutil.rmtree(dirpath)
                else:
                    os.rmdir(dirpath)

            await asyncio.to_thread(_delete)
            return {"success": True, "path": dirpath, "status": "deleted"}
        except OSError as e:
            if "not empty" in str(e).lower():
                return {
                    "success": False,
                    "error": "DirectoryNotEmpty",
                    "message": f"Directory not empty: {dirpath}",
                }
            return {"success": False, "error": type(e).__name__, "message": str(e)}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def tree_view(
        dirpath: str = ".", max_depth: int = 3, show_hidden: bool = False
    ) -> Dict[str, Any]:
        """
        Generate tree view of directory structure.

        Args:
            dirpath (str): Root directory path
            max_depth (int): Maximum depth to traverse
            show_hidden (bool): Show hidden files

        Returns:
            Dict with success status, directory, tree string, and line count
        """
        try:

            def _build_tree(path: Path, prefix: str = "", current_depth: int = 0) -> List[str]:
                if current_depth > max_depth:
                    return []

                lines = []
                try:
                    items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                    items = [i for i in items if show_hidden or not i.name.startswith(".")]

                    for i, item in enumerate(items):
                        is_last = i == len(items) - 1
                        connector = "└── " if is_last else "├── "

                        if item.is_dir():
                            lines.append(f"{prefix}{connector}{item.name}/")
                            extension = "    " if is_last else "│   "
                            lines.extend(_build_tree(item, prefix + extension, current_depth + 1))
                        else:
                            size = item.stat().st_size
                            size_str = f" ({size} bytes)" if size < 1024 else f" ({size//1024} KB)"
                            lines.append(f"{prefix}{connector}{item.name}{size_str}")
                except PermissionError:
                    lines.append(f"{prefix}[Permission Denied]")

                return lines

            path = Path(dirpath)
            if not path.exists():
                return {
                    "success": False,
                    "error": "FileNotFoundError",
                    "message": f"Directory not found: {dirpath}",
                }

            tree_lines = [f"{path.name}/"] + await asyncio.to_thread(_build_tree, path)
            tree_str = "\n".join(tree_lines)

            return {
                "success": True,
                "directory": dirpath,
                "tree": tree_str,
                "lines": len(tree_lines),
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def find_files(
        dirpath: str,
        name_pattern: str = "*",
        extension: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Search for files matching criteria.

        Args:
            dirpath (str): Directory to search
            name_pattern (str): Filename pattern
            extension (str): Filter by file extension
            min_size (int): Minimum file size in bytes
            max_size (int): Maximum file size in bytes
            recursive (bool): Search recursively

        Returns:
            Dict with success status, files list, and count
        """
        try:

            def _find():
                path = Path(dirpath)
                glob_pattern = f"**/{name_pattern}" if recursive else name_pattern
                results = []

                for file in path.glob(glob_pattern):
                    if not file.is_file():
                        continue

                    if extension and file.suffix != extension:
                        continue

                    try:
                        size = file.stat().st_size
                        if min_size and size < min_size:
                            continue
                        if max_size and size > max_size:
                            continue

                        results.append(
                            {
                                "name": file.name,
                                "path": str(file.absolute()),
                                "size": size,
                                "extension": file.suffix,
                            }
                        )
                    except Exception:
                        continue

                return results

            files = await asyncio.to_thread(_find)
            return {
                "success": True,
                "directory": dirpath,
                "files": files,
                "count": len(files),
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def get_directory_size(dirpath: str) -> Dict[str, Any]:
        """
        Calculate total size of directory and its contents.

        Args:
            dirpath (str): Directory path

        Returns:
            Dict with success status, size in bytes, MB, and GB
        """
        try:

            def _get_size():
                total = 0
                for item in Path(dirpath).rglob("*"):
                    if item.is_file():
                        try:
                            total += item.stat().st_size
                        except Exception:
                            continue
                return total

            size = await asyncio.to_thread(_get_size)
            return {
                "success": True,
                "directory": dirpath,
                "size": size,
                "size_mb": round(size / (1024 * 1024), 2),
                "size_gb": round(size / (1024 * 1024 * 1024), 2),
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}


class SearchOperations:
    """Text search operations using regex patterns in files and directories."""

    @staticmethod
    async def grep_in_file(
        filepath: str,
        pattern: str,
        case_sensitive: bool = True,
        max_results: int = 1000,
        context_lines: int = 0,
    ) -> Dict[str, Any]:
        """
        Search for regex pattern in a file.

        Args:
            filepath (str): Path to file to search
            pattern (str): Regex pattern to search for
            case_sensitive (bool): Case sensitive search
            max_results (int): Maximum number of matches to return
            context_lines (int): Number of context lines before/after match

        Returns:
            Dict with success status, matches, and count
        """
        try:

            def _grep():
                flags = 0 if case_sensitive else re.IGNORECASE
                regex = re.compile(pattern, flags)
                results = []

                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                    for line_num, line in enumerate(lines, 1):
                        if regex.search(line):
                            result = {
                                "line_number": line_num,
                                "content": line.rstrip(),
                                "file": filepath,
                            }

                            if context_lines > 0:
                                start = max(0, line_num - context_lines - 1)
                                end = min(len(lines), line_num + context_lines)
                                result["context"] = [l.rstrip() for l in lines[start:end]]

                            results.append(result)

                            if len(results) >= max_results:
                                break

                return results

            matches = await asyncio.to_thread(_grep)
            return {
                "success": True,
                "file": filepath,
                "pattern": pattern,
                "matches": matches,
                "count": len(matches),
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def grep_in_directory(
        dirpath: str,
        pattern: str,
        file_pattern: str = "*",
        recursive: bool = True,
        case_sensitive: bool = True,
        max_results: int = 1000,
    ) -> Dict[str, Any]:
        """
        Search for regex pattern in all files in directory.

        Args:
            dirpath (str): Directory to search
            pattern (str): Regex pattern to search for
            file_pattern (str): File glob pattern to filter files
            recursive (bool): Search recursively
            case_sensitive (bool): Case sensitive search
            max_results (int): Maximum number of matches to return

        Returns:
            Dict with success status, matches list, and count
        """
        try:

            def _grep():
                path = Path(dirpath)
                glob_pattern = f"**/{file_pattern}" if recursive else file_pattern
                all_results = []

                for file in path.glob(glob_pattern):
                    if not file.is_file():
                        continue

                    try:
                        flags = 0 if case_sensitive else re.IGNORECASE
                        regex = re.compile(pattern, flags)

                        with open(file, "r", encoding="utf-8", errors="ignore") as f:
                            for line_num, line in enumerate(f, 1):
                                if regex.search(line):
                                    all_results.append(
                                        {
                                            "file": str(file.absolute()),
                                            "line_number": line_num,
                                            "content": line.rstrip(),
                                        }
                                    )

                                    if len(all_results) >= max_results:
                                        return all_results
                    except Exception:
                        continue

                return all_results

            matches = await asyncio.to_thread(_grep)
            return {
                "success": True,
                "directory": dirpath,
                "pattern": pattern,
                "matches": matches,
                "count": len(matches),
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}


class CommandExecutor:
    """Execute system commands and shell scripts asynchronously."""

    @staticmethod
    async def run_command(
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 30,
        shell: bool = True,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute system command asynchronously.

        Args:
            command (str): Command to execute
            cwd (str): Working directory for command
            timeout (int): Timeout in seconds
            shell (bool): Execute through shell
            env (dict): Environment variables

        Returns:
            Dict with success status, return code, stdout, and stderr
        """
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)

                return {
                    "success": process.returncode == 0,
                    "returncode": process.returncode,
                    "stdout": stdout.decode("utf-8", errors="ignore"),
                    "stderr": stderr.decode("utf-8", errors="ignore"),
                    "command": command,
                }
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": "TimeoutError",
                    "message": f"Command timed out after {timeout}s",
                    "command": command,
                }
        except Exception as e:
            return {
                "success": False,
                "error": type(e).__name__,
                "message": str(e),
                "command": command,
            }

    @staticmethod
    async def run_shell_script(
        script: str, cwd: Optional[str] = None, timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Execute shell script.

        Args:
            script (str): Shell script content
            cwd (str): Working directory
            timeout (int): Timeout in seconds

        Returns:
            Dict with execution results
        """
        return await CommandExecutor.run_command(script, cwd=cwd, timeout=timeout, shell=True)

    @staticmethod
    async def kill_process(pid: int, signal: int = 9) -> Dict[str, Any]:
        """
        Kill a process by PID.

        Args:
            pid (int): Process ID to kill
            signal (int): Signal number (default: 9 for SIGKILL)

        Returns:
            Dict with success status and kill confirmation
        """
        try:

            def _kill():
                os.kill(pid, signal)

            await asyncio.to_thread(_kill)
            return {"success": True, "pid": pid, "signal": signal, "status": "killed"}
        except ProcessLookupError:
            return {
                "success": False,
                "error": "ProcessLookupError",
                "message": f"Process not found: {pid}",
            }
        except PermissionError:
            return {
                "success": False,
                "error": "PermissionError",
                "message": f"Permission denied for PID: {pid}",
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}


class PythonExecutor:
    """Execute and evaluate Python code in isolated context."""

    @staticmethod
    async def exec_python_code(
        code: str, timeout: int = 10, capture_output: bool = True
    ) -> Dict[str, Any]:
        """
        Execute Python code in isolated context.

        Args:
            code (str): Python code to execute
            timeout (int): Execution timeout in seconds
            capture_output (bool): Capture stdout and stderr

        Returns:
            Dict with success status, stdout, stderr, and local variables
        """
        try:
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr

            async def _exec():
                if capture_output:
                    stdout_buffer = io.StringIO()
                    stderr_buffer = io.StringIO()

                    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                        local_vars = {}
                        exec(code, {}, local_vars)

                    return (
                        stdout_buffer.getvalue(),
                        stderr_buffer.getvalue(),
                        local_vars,
                    )
                else:
                    local_vars = {}
                    exec(code, {}, local_vars)
                    return "", "", local_vars

            stdout, stderr, local_vars = await asyncio.wait_for(_exec(), timeout=timeout)

            return {
                "success": True,
                "stdout": stdout,
                "stderr": stderr,
                "locals": {k: str(v) for k, v in local_vars.items() if not k.startswith("__")},
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "TimeoutError",
                "message": f"Execution timed out after {timeout}s",
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def exec_python_file(filepath: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute Python file.

        Args:
            filepath (str): Path to Python file
            timeout (int): Execution timeout in seconds

        Returns:
            Dict with execution results
        """
        try:

            def _read():
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()

            code = await asyncio.to_thread(_read)
            return await PythonExecutor.exec_python_code(code, timeout)
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def eval_python_expr(expression: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Evaluate Python expression and return result.

        Args:
            expression (str): Python expression to evaluate
            timeout (int): Evaluation timeout in seconds

        Returns:
            Dict with success status, result, result string, and type
        """
        try:

            async def _eval():
                return eval(expression, {"__builtins__": __builtins__}, {})

            result = await asyncio.wait_for(_eval(), timeout=timeout)
            return {
                "success": True,
                "result": result,
                "result_str": str(result),
                "type": type(result).__name__,
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "TimeoutError",
                "message": f"Evaluation timed out after {timeout}s",
            }
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}


file_ops = FileOperations()
dir_ops = DirectoryOperations()
search_ops = SearchOperations()
cmd_exec = CommandExecutor()
py_exec = PythonExecutor()


if __name__ == "__main__":

    async def test_all():
        print("=" * 80)
        print("Testing Utils Library")
        print("=" * 80)
        print()

        test_dir = "./test_utils_temp"
        test_file = f"{test_dir}/test.txt"

        print("=" * 80)
        print("1. Testing FileOperations")
        print("=" * 80)

        await dir_ops.create_directory(test_dir)
        print(f"✓ Created directory: {test_dir}")

        write_result = await file_ops.write_file(test_file, "Hello World\nLine 2\nLine 3\n")
        print(f"✓ Write file: {write_result}")

        read_result = await file_ops.read_file(test_file)
        print(f"✓ Read file: success={read_result['success']}, size={read_result.get('size', 0)}")

        lines_result = await file_ops.read_file_lines(test_file, 1, 2)
        print(f"✓ Read lines 1-2: {lines_result['total_lines']} lines")

        append_result = await file_ops.append_file(test_file, "Line 4\n")
        print(f"✓ Append file: new_size={append_result.get('new_size', 0)}")

        info_result = await file_ops.get_file_info(test_file)
        print(
            f"✓ File info: size={info_result.get('size', 0)}, extension={info_result.get('extension', '')}"
        )

        replace_result = await file_ops.replace_in_file(test_file, "Line", "Row")
        print(f"✓ Replace in file: {replace_result['replaced']} replacements")

        exists_result = await file_ops.file_exists(test_file)
        print(f"✓ File exists: {exists_result['exists']}")

        copy_result = await file_ops.copy_file(test_file, f"{test_dir}/test_copy.txt")
        print(f"✓ Copy file: {copy_result['status']}")

        move_result = await file_ops.move_file(
            f"{test_dir}/test_copy.txt", f"{test_dir}/test_moved.txt"
        )
        print(f"✓ Move file: {move_result['status']}")

        print()
        print("=" * 80)
        print("2. Testing DirectoryOperations")
        print("=" * 80)

        subdir_result = await dir_ops.create_directory(f"{test_dir}/subdir")
        print(f"✓ Create subdirectory: {subdir_result['status']}")

        await file_ops.write_file(f"{test_dir}/subdir/file1.txt", "content1")
        await file_ops.write_file(f"{test_dir}/subdir/file2.py", "print('hello')")
        print(f"✓ Created test files in subdirectory")

        list_result = await dir_ops.list_directory(test_dir, recursive=False)
        print(f"✓ List directory: {list_result['count']} items")
        for item in list_result["items"]:
            print(f"  - {item['name']} ({item['type']})")

        tree_result = await dir_ops.tree_view(test_dir, max_depth=2)
        print(f"✓ Tree view:")
        print(tree_result["tree"])

        find_result = await dir_ops.find_files(test_dir, "*.txt", recursive=True)
        print(f"✓ Find files: {find_result['count']} .txt files")
        for f in find_result["files"]:
            print(f"  - {f['name']}")

        size_result = await dir_ops.get_directory_size(test_dir)
        print(f"✓ Directory size: {size_result['size']} bytes ({size_result['size_mb']} MB)")

        print()
        print("=" * 80)
        print("3. Testing SearchOperations")
        print("=" * 80)

        await file_ops.write_file(
            f"{test_dir}/search_test.txt",
            "TODO: fix this\nNormal line\nTODO: another task\n",
        )

        grep_file_result = await search_ops.grep_in_file(f"{test_dir}/search_test.txt", "TODO")
        print(f"✓ Grep in file: {grep_file_result['count']} matches")
        for match in grep_file_result["matches"]:
            print(f"  Line {match['line_number']}: {match['content']}")

        grep_dir_result = await search_ops.grep_in_directory(
            test_dir, "TODO", file_pattern="*.txt", recursive=True
        )
        print(f"✓ Grep in directory: {grep_dir_result['count']} matches")

        grep_case_result = await search_ops.grep_in_file(
            f"{test_dir}/search_test.txt", "todo", case_sensitive=False
        )
        print(f"✓ Case-insensitive grep: {grep_case_result['count']} matches")

        print()
        print("=" * 80)
        print("4. Testing CommandExecutor")
        print("=" * 80)

        import platform

        if platform.system() == "Windows":
            cmd_result = await cmd_exec.run_command("dir", cwd=test_dir, timeout=5)
            print(f"✓ Run command 'dir': success={cmd_result['success']}")
            print(f"  Output preview: {cmd_result['stdout'][:100]}...")

            echo_result = await cmd_exec.run_command("echo Hello from command", timeout=5)
            print(f"✓ Run echo command: {echo_result['stdout'].strip()}")
        else:
            cmd_result = await cmd_exec.run_command("ls -la", cwd=test_dir, timeout=5)
            print(f"✓ Run command 'ls -la': success={cmd_result['success']}")
            print(f"  Output preview: {cmd_result['stdout'][:100]}...")

            echo_result = await cmd_exec.run_command('echo "Hello from command"', timeout=5)
            print(f"✓ Run echo command: {echo_result['stdout'].strip()}")

        timeout_result = await cmd_exec.run_command("ping 127.0.0.1", timeout=1)
        print(f"✓ Timeout test: error={timeout_result.get('error', 'none')}")

        print()
        print("=" * 80)
        print("5. Testing PythonExecutor")
        print("=" * 80)

        exec_result = await py_exec.exec_python_code(
            "print('Hello from Python')\nx = 42\ny = x * 2", timeout=5
        )
        print(f"✓ Exec Python code: success={exec_result['success']}")
        print(f"  stdout: {exec_result['stdout'].strip()}")
        print(f"  locals: {exec_result['locals']}")

        eval_result = await py_exec.eval_python_expr("2 + 3 * 4", timeout=5)
        print(f"✓ Eval expression: result={eval_result['result']}, type={eval_result['type']}")

        await file_ops.write_file(
            f"{test_dir}/test_script.py", "print('Script executed')\nresult = 100 + 200"
        )
        exec_file_result = await py_exec.exec_python_file(f"{test_dir}/test_script.py", timeout=5)
        print(f"✓ Exec Python file: success={exec_file_result['success']}")
        print(f"  stdout: {exec_file_result['stdout'].strip()}")

        error_result = await py_exec.exec_python_code("1/0", timeout=5)
        print(
            f"✓ Error handling: success={error_result['success']}, error={error_result.get('error', 'none')}"
        )

        print()
        print("=" * 80)
        print("6. Testing Error Handling")
        print("=" * 80)

        nonexist_read = await file_ops.read_file("nonexistent_file.txt")
        print(
            f"✓ Read nonexistent file: success={nonexist_read['success']}, error={nonexist_read.get('error', 'none')}"
        )

        nonexist_list = await dir_ops.list_directory("nonexistent_dir")
        print(
            f"✓ List nonexistent dir: success={nonexist_list['success']}, error={nonexist_list.get('error', 'none')}"
        )

        invalid_grep = await search_ops.grep_in_file("nonexistent.txt", "pattern")
        print(
            f"✓ Grep nonexistent file: success={invalid_grep['success']}, error={invalid_grep.get('error', 'none')}"
        )

        print()
        print("=" * 80)
        print("7. Cleanup")
        print("=" * 80)

        delete_file_result = await file_ops.delete_file(test_file)
        print(f"✓ Delete test file: {delete_file_result['status']}")

        delete_dir_result = await dir_ops.delete_directory(test_dir, recursive=True)
        print(f"✓ Delete test directory: {delete_dir_result['status']}")

        print()
        print("=" * 80)
        print("All Tests Completed!")
        print("=" * 80)

    asyncio.run(test_all())
