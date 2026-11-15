import logging
import os
import re
import shlex
import subprocess
from abc import ABC, abstractmethod

from cellink.resources._utils import get_data_home

logger = logging.getLogger(__name__)


def _is_within(path: str, root: str) -> bool:
    """Whether `path` is `root` or lies below it (``/a/gwas_data`` is not within ``/a/gwas``)."""
    path, root = os.path.abspath(path), os.path.abspath(root)
    return os.path.commonpath([path, root]) == root


def _replace_host_paths(command: str, volumes: dict[str, str]) -> str:
    """Replace host directories in `command` by their container paths, at path boundaries only."""
    for host_path, container_path in volumes.items():
        host_path = os.path.abspath(str(host_path))
        pattern = rf"(?<![\w./-]){re.escape(host_path)}(?=/|[\s'\",;:]|$)"
        command = re.sub(pattern, lambda _, c=str(container_path): c, command)
    return command


class BaseToolRunner(ABC):
    """BaseToolRunner with YAML config and automatic path inference"""

    def __init__(
        self,
        config_path: str | None = None,
        config_dict: dict | None = None,
        required_fields: list | None = None,
        prefix_tokens: list | None = None,
    ):
        """Initialize Runner.

        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file
        config_dict : dict, optional
            Configuration dictionary (takes precedence over config_path)
        """
        self.required_fields = required_fields if required_fields is not None else ["execution_mode"]
        self.prefix_tokens = prefix_tokens if prefix_tokens is not None else []
        self.config = self._load_config(config_path, config_dict)
        self._validate_config()

    @abstractmethod
    def _load_config(self, config_path: str | None, config_dict: dict | None) -> dict:
        """Load configuration from file or dictionary.

        Must be implemented by subclasses.
        """
        pass

    def _validate_config(self):
        """Validate configuration parameters"""
        for field in self.required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        if self.config["execution_mode"] not in ["local", "docker", "singularity"]:
            raise ValueError("execution_mode must be 'local', 'docker', or 'singularity'")

    def _infer_volumes_from_paths(self, *file_paths: str, data_home: str | None = None) -> dict[str, str]:
        """
        Automatically infer docker volumes or singularity binds from file paths

        Parameters
        ----------
        *file_paths : str
            Variable number of file paths to analyze

        Returns
        -------
        dict
            Dictionary mapping host paths to container paths
        """
        volumes = {}

        volumes[os.getcwd()] = "/data"

        cellink_data_path = str(get_data_home(data_home))
        if os.path.exists(cellink_data_path):
            volumes[cellink_data_path] = "/cellink_data"

        for file_path in file_paths:
            if file_path:
                abs_path = os.path.abspath(file_path)
                parent_dir = os.path.dirname(abs_path)

                covered = any(_is_within(abs_path, str(host_path)) for host_path in volumes)

                if not covered:
                    container_path = f"/external_{len(volumes)}"
                    volumes[parent_dir] = container_path

        return volumes

    def _convert_path_to_container(self, file_path: str, volumes: dict[str, str]) -> str:
        """Convert host path to container path"""
        if not file_path:
            return file_path

        abs_path = os.path.abspath(file_path)

        for host_path, container_path in volumes.items():
            host_path = str(host_path)
            if _is_within(abs_path, host_path):
                relative_path = os.path.relpath(abs_path, host_path)
                return os.path.join(container_path, relative_path).replace("\\", "/")

        return file_path

    def _build_container_command(self, base_command: str, file_paths: list[str] = None) -> str:
        """Build docker or singularity command with volumes"""
        if self.config["execution_mode"] == "local":
            return base_command

        if file_paths is None:
            file_paths = []

        volumes = self._infer_volumes_from_paths(*file_paths)
        container_command = self._rewrite_paths_in_command(base_command, volumes)

        # host paths inside tokens such as --bedFile=/abs/path.bed
        container_command = _replace_host_paths(container_command, volumes)

        if self.config["execution_mode"] == "docker":
            volume_args = []
            for host_path, container_path in volumes.items():
                volume_args.extend(["-v", f"{host_path}:{container_path}"])

            cmd = ["docker", "run", "--rm", *volume_args, "-w", "/data", self.config["docker_image"]]
            return f"{shlex.join(cmd)} {container_command}"

        elif self.config["execution_mode"] == "singularity":
            bind_args = []
            for host_path, container_path in volumes.items():
                bind_args.extend(["-B", f"{host_path}:{container_path}"])

            image = self.config.get("singularity_image", "")
            # like docker's `-w /data`: relative paths in the command refer to the bound working directory
            cmd = ["singularity", "exec", *bind_args, "--pwd", "/data", image]

            return f"{shlex.join(cmd)} {container_command}"

        return base_command

    def _rewrite_one_path(self, path: str, volumes: dict[str, str]) -> str:
        """Rewrite a single host path/prefix to its container-side equivalent, if covered by `volumes`."""
        abs_path = os.path.abspath(path)
        for host_path, container_path in volumes.items():
            if _is_within(abs_path, host_path):
                rel = os.path.relpath(abs_path, host_path)
                return os.path.join(container_path, rel).replace("\\", "/")
        return path

    def _rewrite_paths_in_command(self, command: str, volumes: dict[str, str]) -> str:
        tokens = shlex.split(command)
        rewritten = []

        for token_i, token in enumerate(tokens):
            new_token = token
            after_prefix_token = token_i > 1 and tokens[token_i - 1] in self.prefix_tokens

            if os.path.exists(token) or after_prefix_token:
                if after_prefix_token and "," in token:
                    new_token = ",".join(self._rewrite_one_path(part, volumes) for part in token.split(","))
                else:
                    new_token = self._rewrite_one_path(token, volumes)

            rewritten.append(new_token)

        return shlex.join(rewritten)

    def run_command(self, base_command: str, file_paths: list[str] = None, check: bool = True):
        """
        Execute command with automatic path inference

        The command is split with :func:`shlex.split` and run without a shell, so file names are
        never interpreted by a shell; callers quote arguments with :func:`shlex.quote`.

        Parameters
        ----------
        base_command : str
            The base LDSC command
        file_paths : list, optional
            List of file paths involved in the command (for volume inference)
        check : bool
            Whether to raise exception on command failure
        """
        if file_paths is None:
            file_paths = []
        if os.getcwd() not in file_paths:
            file_paths.append(os.getcwd())

        if self.config["execution_mode"] == "local":
            result = subprocess.run(shlex.split(base_command), check=False, capture_output=True, text=True)
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, base_command, result.stdout, result.stderr)
        else:
            full_command = self._build_container_command(base_command, file_paths)

            logger.info(f"Executing: {full_command}")
            result = subprocess.run(shlex.split(full_command), check=False, capture_output=True, text=True)
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, full_command, result.stdout, result.stderr)
