import logging
import os
import shlex
import subprocess
from abc import ABC, abstractmethod

from cellink.resources._utils import get_data_home

logger = logging.getLogger(__name__)


class BaseToolRunner(ABC):
    """BaseToolRunner with YAML config and automatic path inference"""

    def __init__(
        self,
        config_path: str | None = None,
        config_dict: dict | None = None,
        required_fields: list = ["execution_mode"],
        prefix_tokens: list = [],
    ):
        """
        Initialize Runner

        Parameters
        ----------
        config_path : str, optional
            Path to YAML configuration file
        config_dict : dict, optional
            Configuration dictionary (takes precedence over config_path)
        """
        self.required_fields = required_fields
        self.prefix_tokens = prefix_tokens
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

                covered = False
                for host_path in volumes.keys():
                    host_path = str(host_path)
                    if abs_path.startswith(host_path):
                        covered = True
                        break

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
            if abs_path.startswith(host_path):
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

        # container_command = base_command
        for host_path, container_path in volumes.items():
            container_command = str(container_command).replace(str(host_path), str(container_path))

        if self.config["execution_mode"] == "docker":
            volume_args = []
            for host_path, container_path in volumes.items():
                volume_args.extend(["-v", f"{host_path}:{container_path}"])

            cmd = ["docker", "run", "--rm", *volume_args, "-w", "/data", self.config["docker_image"], container_command]
            return " ".join(cmd)

        elif self.config["execution_mode"] == "singularity":
            bind_args = []
            for host_path, container_path in volumes.items():
                bind_args.extend(["-B", f"{host_path}:{container_path}"])

            cmd = ["singularity", "exec", *bind_args, self.config["singularity_image"], container_command]
            return " ".join(cmd)

        return base_command

    def _rewrite_paths_in_command(self, command: str, volumes: dict[str, str]) -> str:
        tokens = shlex.split(command)
        rewritten = []

        for token_i, token in enumerate(tokens):
            new_token = token

            if os.path.exists(token) or (token_i > 1 and tokens[token_i - 1] in self.prefix_tokens):
                abs_path = os.path.abspath(token)

                for host_path, container_path in volumes.items():
                    if abs_path.startswith(host_path):
                        rel = os.path.relpath(abs_path, host_path)
                        new_token = os.path.join(container_path, rel).replace("\\", "/")
                        break

            rewritten.append(new_token)

        return " ".join(rewritten)

    def run_command(self, base_command: str, file_paths: list[str] = None, check: bool = True):
        """
        Execute command with automatic path inference

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
            result = subprocess.run(base_command, shell=True, check=check, capture_output=True, text=True)
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        else:
            full_command = self._build_container_command(base_command, file_paths)

            logger.info(f"Executing: {full_command}")
            result = subprocess.run(full_command, shell=True, check=check, capture_output=True, text=True)
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
