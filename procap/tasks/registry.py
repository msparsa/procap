"""
Task registry for ProCap benchmark.

Provides a unified interface for loading and managing benchmark tasks.

Usage:
    # Get a task by name
    task = get_task("go_term_identification")

    # List tasks by Bloom level
    tasks = TaskRegistry().filter_by_bloom_level(BloomLevel.REMEMBERING)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from procap.tasks.schemas import BloomLevel, TaskConfig, TaskType, DEFAULT_TASKS


class TaskRegistry:
    """
    Registry for benchmark tasks.

    Manages task configurations and provides filtering capabilities.

    Usage:
        registry = TaskRegistry()

        # Get a task
        task = registry.get("go_term_identification")

        # Filter tasks
        remembering_tasks = registry.filter_by_bloom_level(BloomLevel.REMEMBERING)
        model_tasks = registry.filter_by_model("esm2_t33_650M")
    """

    def __init__(self, load_defaults: bool = True):
        """
        Initialize task registry.

        Args:
            load_defaults: Whether to load default task configurations
        """
        self._tasks: Dict[str, TaskConfig] = {}

        if load_defaults:
            for name, config in DEFAULT_TASKS.items():
                self._tasks[name] = TaskConfig(**config)

    def register(
        self,
        task: TaskConfig,
        overwrite: bool = False,
    ) -> None:
        """
        Register a task configuration.

        Args:
            task: Task configuration
            overwrite: Whether to overwrite existing registration
        """
        if task.name in self._tasks and not overwrite:
            raise ValueError(
                f"Task {task.name!r} already registered. Use overwrite=True to replace."
            )

        self._tasks[task.name] = task

    def get(self, name: str) -> TaskConfig:
        """
        Get a task by name.

        Args:
            name: Task identifier

        Returns:
            Task configuration

        Raises:
            ValueError: If task not found
        """
        if name not in self._tasks:
            available = ", ".join(self.list_tasks())
            raise ValueError(f"Unknown task {name!r}. Available tasks: {available}")

        return self._tasks[name]

    def list_tasks(self) -> List[str]:
        """List all registered task names."""
        return list(self._tasks.keys())

    def filter_by_bloom_level(self, level: BloomLevel) -> List[TaskConfig]:
        """
        Get all tasks at a specific Bloom taxonomy level.

        Args:
            level: Bloom taxonomy level

        Returns:
            List of matching task configurations
        """
        return [t for t in self._tasks.values() if t.bloom_level == level]

    def filter_by_model(self, model_name: str) -> List[TaskConfig]:
        """
        Get all tasks compatible with a specific model.

        Args:
            model_name: Model identifier

        Returns:
            List of compatible task configurations
        """
        return [
            t
            for t in self._tasks.values()
            if model_name in t.compatible_models or not t.compatible_models
        ]

    def filter_by_domain(self, domain: str) -> List[TaskConfig]:
        """
        Get all tasks in a specific domain.

        Args:
            domain: Domain name

        Returns:
            List of matching task configurations
        """
        return [t for t in self._tasks.values() if t.domain == domain]

    def filter_by_task_type(self, task_type: TaskType) -> List[TaskConfig]:
        """
        Get all tasks of a specific type.

        Args:
            task_type: Task type

        Returns:
            List of matching task configurations
        """
        return [t for t in self._tasks.values() if t.task_type == task_type]

    def get_all(self) -> List[TaskConfig]:
        """Get all registered tasks."""
        return list(self._tasks.values())

    def load_from_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Load task configurations from a YAML file.

        YAML format:
            tasks:
              - name: my_task
                bloom_level: Remembering
                domain: Functional Annotation
                description: My task description
                task_type: multilabel_classification
                dataset_path: data/my_task.csv
                input_fields: [sequence]
                target_field: labels
                metrics: [f1_max, auprc_micro]
                compatible_models: [esm2_t33_650M]
        """
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for task_dict in config.get("tasks", []):
            # Convert string enum values
            if "bloom_level" in task_dict and isinstance(task_dict["bloom_level"], str):
                task_dict["bloom_level"] = BloomLevel(task_dict["bloom_level"])
            if "task_type" in task_dict and isinstance(task_dict["task_type"], str):
                task_dict["task_type"] = TaskType(task_dict["task_type"])

            task = TaskConfig(**task_dict)
            self.register(task, overwrite=True)

    def load_from_directory(self, config_dir: Union[str, Path]) -> None:
        """
        Load all YAML files from a directory.

        Args:
            config_dir: Directory containing YAML files
        """
        config_dir = Path(config_dir)
        for yaml_file in config_dir.glob("*.yaml"):
            self.load_from_yaml(yaml_file)
        for yaml_file in config_dir.glob("*.yml"):
            self.load_from_yaml(yaml_file)

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of registered tasks.

        Returns:
            Dictionary with counts by Bloom level, domain, and task type
        """
        by_bloom = {}
        for level in BloomLevel:
            tasks = self.filter_by_bloom_level(level)
            by_bloom[level.value] = len(tasks)

        domains = set(t.domain for t in self._tasks.values())
        by_domain = {d: len(self.filter_by_domain(d)) for d in domains}

        by_type = {}
        for tt in TaskType:
            tasks = self.filter_by_task_type(tt)
            if tasks:
                by_type[tt.value] = len(tasks)

        return {
            "total": len(self._tasks),
            "by_bloom_level": by_bloom,
            "by_domain": by_domain,
            "by_task_type": by_type,
        }


# Global registry instance
_global_registry = TaskRegistry(load_defaults=False)

# Load tasks from YAML config
_config_path = Path(__file__).parent.parent.parent / "configs" / "tasks.yaml"
if _config_path.exists():
    _global_registry.load_from_yaml(_config_path)


def get_task(name: str) -> TaskConfig:
    """
    Get a task from the global registry.

    Args:
        name: Task identifier

    Returns:
        Task configuration
    """
    return _global_registry.get(name)


def list_tasks() -> List[str]:
    """List all available tasks."""
    return _global_registry.list_tasks()


def register_task(task: TaskConfig) -> None:
    """Register a task in the global registry."""
    _global_registry.register(task)


def get_tasks_by_bloom_level(level: BloomLevel) -> List[TaskConfig]:
    """Get all tasks at a specific Bloom level."""
    return _global_registry.filter_by_bloom_level(level)
