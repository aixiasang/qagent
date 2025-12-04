import os
import re
import asyncio
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


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
            return {"success": True, "content": content, "filepath": filepath, "size": len(content)}
        except FileNotFoundError:
            return {"success": False, "error": "FileNotFoundError", "message": f"File not found: {filepath}"}
        except PermissionError:
            return {"success": False, "error": "PermissionError", "message": f"Permission denied: {filepath}"}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def write_file(filepath: str, content: str, encoding: str = "utf-8", create_dirs: bool = True) -> Dict[str, Any]:
        """
        Write content to file.

        Args:
            filepath (str): Path to the file
            content (str): Content to write
            encoding (str): File encoding
            create_dirs (bool): Create parent directories if needed
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
        """Append content to existing file."""
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
        """Delete a file."""
        try:
            if not os.path.exists(filepath):
                return {"success": False, "error": "FileNotFoundError", "message": f"File not found: {filepath}"}
            if os.path.isdir(filepath):
                return {"success": False, "error": "IsADirectoryError", "message": f"Path is a directory: {filepath}"}

            await asyncio.to_thread(os.remove, filepath)
            return {"success": True, "filepath": filepath, "status": "deleted"}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def file_exists(filepath: str) -> Dict[str, Any]:
        """Check if file exists."""
        try:
            exists = await asyncio.to_thread(os.path.exists, filepath)
            return {"success": True, "exists": exists, "filepath": filepath}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def get_file_info(filepath: str) -> Dict[str, Any]:
        """Get detailed file information."""
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
            return {"success": False, "error": "FileNotFoundError", "message": f"Path not found: {filepath}"}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}


class DirectoryOperations:

    @staticmethod
    async def list_directory(
        dirpath: str = ".",
        recursive: bool = False,
        show_hidden: bool = False,
        pattern: str = "*",
    ) -> Dict[str, Any]:
        """
        List files and directories.

        Args:
            dirpath (str): Directory path
            recursive (bool): Include subdirectories
            show_hidden (bool): Show hidden files
            pattern (str): Glob pattern
        """
        try:
            def _list():
                path = Path(dirpath)
                if not path.exists():
                    raise FileNotFoundError(f"Directory not found: {dirpath}")

                glob_pattern = f"**/{pattern}" if recursive else pattern
                results = []

                for item in path.glob(glob_pattern):
                    if not show_hidden and item.name.startswith("."):
                        continue
                    try:
                        stat = item.stat()
                        results.append({
                            "name": item.name,
                            "path": str(item.absolute()),
                            "type": "dir" if item.is_dir() else "file",
                            "size": stat.st_size if item.is_file() else None,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        })
                    except Exception:
                        continue
                return results

            items = await asyncio.to_thread(_list)
            return {"success": True, "directory": dirpath, "items": items, "count": len(items)}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def create_directory(dirpath: str, parents: bool = True, exist_ok: bool = True) -> Dict[str, Any]:
        """Create a new directory."""
        try:
            await asyncio.to_thread(Path(dirpath).mkdir, parents=parents, exist_ok=exist_ok)
            return {"success": True, "path": dirpath, "status": "created"}
        except FileExistsError:
            return {"success": False, "error": "FileExistsError", "message": f"Directory exists: {dirpath}"}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def delete_directory(dirpath: str, recursive: bool = False) -> Dict[str, Any]:
        """Delete a directory."""
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
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}

    @staticmethod
    async def find_files(
        dirpath: str,
        name_pattern: str = "*",
        extension: Optional[str] = None,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """Search for files matching criteria."""
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
                        results.append({
                            "name": file.name,
                            "path": str(file.absolute()),
                            "size": file.stat().st_size,
                            "extension": file.suffix,
                        })
                    except Exception:
                        continue
                return results

            files = await asyncio.to_thread(_find)
            return {"success": True, "directory": dirpath, "files": files, "count": len(files)}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}


class SearchOperations:

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
            filepath (str): Path to file
            pattern (str): Regex pattern
            case_sensitive (bool): Case sensitive search
            max_results (int): Maximum matches
            context_lines (int): Context lines around match
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
                            result = {"line_number": line_num, "content": line.rstrip(), "file": filepath}

                            if context_lines > 0:
                                start = max(0, line_num - context_lines - 1)
                                end = min(len(lines), line_num + context_lines)
                                result["context"] = [l.rstrip() for l in lines[start:end]]

                            results.append(result)
                            if len(results) >= max_results:
                                break
                return results

            matches = await asyncio.to_thread(_grep)
            return {"success": True, "file": filepath, "pattern": pattern, "matches": matches, "count": len(matches)}
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
        """Search for pattern in all files in directory."""
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
                                    all_results.append({
                                        "file": str(file.absolute()),
                                        "line_number": line_num,
                                        "content": line.rstrip(),
                                    })
                                    if len(all_results) >= max_results:
                                        return all_results
                    except Exception:
                        continue
                return all_results

            matches = await asyncio.to_thread(_grep)
            return {"success": True, "directory": dirpath, "pattern": pattern, "matches": matches, "count": len(matches)}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e)}


class CommandExecutor:

    @staticmethod
    async def run_command(
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 30,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute system command.

        Args:
            command (str): Command to execute
            cwd (str): Working directory
            timeout (int): Timeout in seconds
            env (dict): Environment variables
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
                return {"success": False, "error": "TimeoutError", "message": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": type(e).__name__, "message": str(e), "command": command}
