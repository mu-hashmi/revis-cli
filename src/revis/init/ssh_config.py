"""SSH config parsing utilities."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SSHHost:
    """Parsed SSH host configuration."""

    name: str
    hostname: str
    user: str | None = None
    port: int = 22
    identity_file: str | None = None


def parse_ssh_config() -> list[SSHHost]:
    """Parse ~/.ssh/config and return list of hosts."""
    from paramiko.config import SSHConfig

    config_path = Path.home() / ".ssh" / "config"

    if not config_path.exists():
        return []

    try:
        config = SSHConfig()
        with open(config_path) as f:
            config.parse(f)
    except Exception:
        return []

    hosts = []
    seen_hosts: set[str] = set()

    for entry in config._config:
        host_patterns = entry.get("host", [])
        for pattern in host_patterns:
            if "*" in pattern or "?" in pattern:
                continue
            if pattern in seen_hosts:
                continue
            seen_hosts.add(pattern)

            host_config = config.lookup(pattern)
            hostname = host_config.get("hostname", pattern)

            port = 22
            if "port" in host_config:
                try:
                    port = int(host_config["port"])
                except ValueError:
                    pass

            identity_files = host_config.get("identityfile", [])
            identity_file = identity_files[0] if identity_files else None

            hosts.append(
                SSHHost(
                    name=pattern,
                    hostname=hostname,
                    user=host_config.get("user"),
                    port=port,
                    identity_file=identity_file,
                )
            )

    return hosts
