"""Validación compartida de endpoints de modelos estrictamente locales."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


def is_local_model_endpoint(url: str, allowed_hosts: set[str] | None = None) -> bool:
    """Acepta loopback, RFC1918/ULA o un host configurado explícitamente."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not hostname:
        return False
    if hostname == "localhost" or hostname in (allowed_hosts or set()):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    if address.is_loopback or address.is_link_local:
        return True
    private_networks = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("fc00::/7"),
    )
    return any(address in network for network in private_networks)
