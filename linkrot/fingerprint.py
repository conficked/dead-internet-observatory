"""Text fingerprints that survive the trip from Common Crawl WET text to live HTML.

Common Crawl WET records hold the visible text of a page. A live fetch returns
HTML, which we reduce to visible text the same rough way, then compare 64-bit
SimHash fingerprints of word 3-shingles. Hamming distance between fingerprints
approximates how much of the wording changed.
"""

import hashlib
import re
from html.parser import HTMLParser
from typing import List, Optional, Tuple

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "head"}


def tokens(text: str) -> List[str]:
    return WORD.findall(text.lower())


def _hash64(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def simhash(text: str, shingle: int = 3) -> Optional[int]:
    """64-bit SimHash of word shingles, or None when the text is too short."""
    words = tokens(text)
    if len(words) < shingle + 4:
        return None
    weights = [0] * 64
    for i in range(len(words) - shingle + 1):
        h = _hash64(" ".join(words[i:i + shingle]))
        for bit in range(64):
            weights[bit] += 1 if (h >> bit) & 1 else -1
    value = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            value |= 1 << bit
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def to_hex(value: Optional[int]) -> Optional[str]:
    return None if value is None else f"{value:016x}"


def from_hex(value: Optional[str]) -> Optional[int]:
    return None if not value else int(value, 16)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.title_parts: List[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> Tuple[str, str]:
    """Return (title, visible text) from an HTML document. Never raises."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    title = " ".join(" ".join(parser.title_parts).split())
    text = " ".join(" ".join(parser.parts).split())
    return title[:300], text
