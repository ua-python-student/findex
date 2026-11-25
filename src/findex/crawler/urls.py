"""URL canonicalisation: one spelling per page, so the `seen` set catches duplicates."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_PORTS = {"http": 80, "https": 443}


def canonicalize(url: str, base: str | None = None) -> str | None:
    """Absolute, lowercase scheme/host, no default port, no fragment, sorted query,
    no trailing slash (except the root). Returns None for non-http(s) URLs."""
    if base is not None:
        url = urljoin(base, url)
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return None
    host = parts.hostname.lower()
    if parts.port and parts.port != DEFAULT_PORTS[scheme]:
        host = f"{host}:{parts.port}"
    path = parts.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, host, path, query, ""))


def host_of(url: str) -> str:
    return urlsplit(url).netloc


def domain_allowed(url: str, allowed: frozenset[str]) -> bool:
    """Host (without port) equals an allowed domain or is a subdomain of one."""
    host = (urlsplit(url).hostname or "").lower()
    return not allowed or any(host == d or host.endswith("." + d) for d in allowed)
