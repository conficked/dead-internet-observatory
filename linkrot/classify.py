"""Turn one live fetch of a sampled URL into a fate.

Fates are grouped into four buckets:

    alive    intact, edited, moved, reachable
    drifted  replaced (same address, different page)
    dead     not_found, soft_404, redirect_home, parked, defunct, domain_gone
    unknown  blocked, server_error, unreachable

Content is compared by "retained": the share of a bottom-32 sketch of the original
page's three-word shingles that still appears on the live page. It holds up when a
page gains new navigation or a new design around the same text. Front pages that
still answer count as alive whatever their text, since a front page is meant to
change. Older samples without a sketch fall back to SimHash distance. Thresholds
are heuristics and are published with the results.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

from linkrot.fingerprint import from_hex, hamming, html_to_text, retained, simhash

INTACT_MAX = 10      # SimHash fallback
EDITED_MAX = 22
INTACT_MIN_RETAINED = 0.75
EDITED_MIN_RETAINED = 0.35

BUCKETS = {
    "intact": "alive", "edited": "alive", "moved": "alive", "reachable": "alive",
    "replaced": "drifted",
    "not_found": "dead", "soft_404": "dead", "redirect_home": "dead",
    "parked": "dead", "defunct": "dead", "domain_gone": "dead",
    "blocked": "unknown", "server_error": "unknown", "unreachable": "unknown",
}

BLOCKED_CODES = {401, 403, 407, 429, 451, 999}
GONE_CODES = {404, 410}
ROOT_PATHS = {"", "/", "/index.html", "/index.htm", "/index.php", "/home", "/default.aspx"}

PARKING_HOSTS = (
    "sedoparking.com", "parkingcrew.net", "bodis.com", "above.com", "hugedomains.com",
    "dan.com", "afternic.com", "sedo.com", "undeveloped.com", "domainmarket.com",
    "buydomains.com", "parklogic.com", "domainnamesales.com", "uniregistry.com",
    "namebright.com", "searchvity.com", "smartname.com", "domainlore.com",
)
PARKED_TEXT = re.compile(
    r"(this domain (name )?(is|may be) for sale|buy this domain|domain is parked|"
    r"parked (free|domain)|the domain .{0,60} (is|may be) for sale|"
    r"this domain has (recently )?been registered|make an offer on this domain|"
    r"domain (name )?(is )?available for (purchase|sale)|related searches)",
    re.I,
)
DEFUNCT_TEXT = re.compile(
    r"(welcome to nginx|apache2? (ubuntu|debian)? ?default page|it works!|"
    r"this account has been suspended|account suspended|website (has )?expired|"
    r"site (is )?(temporarily )?(unavailable|offline)|index of /|"
    r"future home of something quite cool|coming soon)",
    re.I,
)
NOT_FOUND_TEXT = re.compile(
    r"(\b404\b|page not found|not be found|could not be found|cannot be found|"
    r"doesn'?t exist|does not exist|no longer (exists|available)|nothing (was )?found|"
    r"page (you|you're) (are )?looking for|has been (removed|deleted))",
    re.I,
)
CHALLENGE_TEXT = re.compile(
    r"(just a moment\.\.\.|attention required!? \| cloudflare|checking your browser|"
    r"enable javascript and cookies to continue|verify you are (a )?human|"
    r"access denied|captcha|ddos protection by)",
    re.I,
)


@dataclass
class Fetch:
    """What came back from one request (after following redirects)."""
    http_status: Optional[int] = None
    final_url: Optional[str] = None
    redirects: int = 0
    content_type: str = ""
    body: str = ""
    error: Optional[str] = None      # dns_nxdomain, dns_temp, timeout, refused, tls, reset, other
    tls_invalid: bool = False
    elapsed_ms: int = 0


@dataclass
class Verdict:
    status: str
    bucket: str
    distance: Optional[int] = None
    retained: Optional[float] = None
    title: str = ""
    notes: list = field(default_factory=list)


def _host(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _site(host: str) -> str:
    """Rough registrable domain: last two labels, three for short second-level labels."""
    labels = host.split(".")
    if len(labels) >= 3 and len(labels[-2]) <= 3 and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _path(url: str) -> str:
    parts = urlsplit(url)
    return (parts.path or "/").rstrip("/") or "/"


ERROR_URL = re.compile(r"(\b404\b|not[-_]?found|pagenotfound|/error\b|errorpage)", re.I)
TITLE_WORD = re.compile(r"[a-z0-9]+")


def title_match(original_first_line: str, title: str) -> bool:
    """True when the live title says what the page's first captured line said."""
    a = set(TITLE_WORD.findall((original_first_line or "").lower()))
    b = set(TITLE_WORD.findall((title or "").lower()))
    if len(a) < 3 or len(b) < 3:
        return False
    return len(a & b) / len(a | b) >= 0.6 or (len(a & b) >= 4 and len(a & b) / len(a) >= 0.8)


def depth(url: str) -> int:
    return len([p for p in urlsplit(url).path.split("/") if p])


def same_location(original: str, final: str) -> bool:
    """True when the redirect only changed scheme, www, or a trailing slash."""
    a, b = urlsplit(original), urlsplit(final)
    return (_host(original) == _host(final) and _path(original) == _path(final)
            and (a.query or "") == (b.query or ""))


def is_root(url: str) -> bool:
    parts = urlsplit(url)
    return (parts.path or "/").lower() in ROOT_PATHS and not parts.query


def verdict(status: str, **kw) -> Verdict:
    return Verdict(status=status, bucket=BUCKETS[status], **kw)


def classify(original_url: str, original_simhash: Optional[str], fetch: Fetch,
             original_sketch: Optional[list] = None, original_first_line: str = "") -> Verdict:
    if fetch.error:
        if fetch.error == "dns_nxdomain":
            return verdict("domain_gone")
        return verdict("unreachable", notes=[fetch.error])

    code = fetch.http_status or 0
    final = fetch.final_url or original_url
    is_html = "html" in fetch.content_type or fetch.body.lstrip()[:15].lower().startswith(("<!doctype", "<html"))
    title, text = html_to_text(fetch.body) if is_html else ("", fetch.body if "text" in fetch.content_type else "")
    head = f"{title} {text[:2000]}"

    if code in BLOCKED_CODES or CHALLENGE_TEXT.search(title):
        return verdict("blocked", title=title)
    if code >= 500:
        return verdict("server_error", title=title)
    if code in GONE_CODES:
        return verdict("not_found", title=title)
    if 400 <= code < 500:
        return verdict("not_found", title=title, notes=[f"http_{code}"])
    if not 200 <= code < 300:
        return verdict("unreachable", title=title, notes=[f"http_{code}"])

    final_host = _host(final)
    if any(final_host == p or final_host.endswith("." + p) for p in PARKING_HOSTS):
        return verdict("parked", title=title)
    if PARKED_TEXT.search(head) and len(text) < 6000:
        return verdict("parked", title=title)

    # How much of the original wording survives
    kept = retained(original_sketch, text) if original_sketch and text else None
    original = from_hex(original_simhash)
    current = simhash(text) if text else None
    distance = hamming(original, current) if original is not None and current is not None else None
    same_title = title_match(original_first_line, title)
    if kept is not None:
        similar = kept >= EDITED_MIN_RETAINED or (same_title and kept >= 0.1)
        level = "intact" if kept >= INTACT_MIN_RETAINED else "edited" if similar else "replaced"
    elif distance is not None:
        similar = distance <= EDITED_MAX or same_title
        level = "intact" if distance <= INTACT_MAX else "edited" if similar else "replaced"
    elif same_title:
        similar, level = True, "edited"
    else:
        similar, level = False, None
    extra = {"distance": distance, "retained": kept, "title": title}

    moved_here = not same_location(original_url, final)
    front = is_root(original_url)
    if not similar and moved_here and ERROR_URL.search(urlsplit(final).path + "?" + urlsplit(final).query):
        return verdict("soft_404", **extra)
    if moved_here and not front and not similar:
        lands_home = (urlsplit(final).path or "/").lower() in ROOT_PATHS
        lands_index = depth(final) <= 1 and depth(original_url) >= 2
        if lands_home or lands_index:
            return verdict("redirect_home", notes=["index"] if not lands_home else [], **extra)

    if not similar:
        if DEFUNCT_TEXT.search(head) and len(text) < 3000:
            return verdict("defunct", **extra)
        if NOT_FOUND_TEXT.search(title) or (NOT_FOUND_TEXT.search(text[:600]) and len(text) < 4000):
            return verdict("soft_404", **extra)

    if front and not similar:
        return verdict("reachable", notes=["front_page"], **extra)
    if level is None:
        if moved_here and _site(final_host) != _site(_host(original_url)):
            return verdict("moved", notes=["offsite_unverified"], **extra)
        return verdict("reachable", notes=["content_not_compared"], **extra)
    if moved_here:
        return verdict("moved" if similar else "replaced", **extra)
    return verdict(level, **extra)
