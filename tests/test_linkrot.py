"""Link-rot census: sampler, fingerprints, classifier, and checker against a local server."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from linkrot import check as checker
from linkrot.classify import Fetch, classify
from linkrot.fingerprint import hamming, html_to_text, simhash, to_hex
from linkrot.sample import last_words, normalize_url, select

ARTICLE = (
    "The county planning board met on Tuesday evening to review three proposals for the "
    "old mill site on the river. Residents raised concerns about traffic on Water Street, "
    "the loss of the historic brick facade, and drainage during spring floods. The first "
    "plan would convert the mill into forty apartments with ground floor retail. The second "
    "keeps the building as a community workshop and market hall run by a nonprofit. The "
    "third, from a regional developer, would demolish the structure and build townhouses. "
    "Board members asked each applicant for a traffic study and a drainage report before "
    "the next meeting in March. A public comment period remains open until the end of the "
    "month, and written comments can be sent to the clerk's office at the town hall."
)
EDITED = ARTICLE.replace("on Tuesday evening", "on Wednesday night").replace(
    "forty apartments", "forty-two apartments").replace("in March", "in early April").replace(
    "The second keeps the building as a community workshop and market hall run by a nonprofit.",
    "Update: the nonprofit behind the second proposal withdrew it after funding fell through.").replace(
    "A public comment period remains open until the end of the month,",
    "Comments closed on the thirtieth after a two week extension,")
OTHER = (
    "Our winter sale is here. Save up to fifty percent on boots, jackets and gloves for the "
    "whole family. Free shipping on orders over seventy five dollars, easy returns within "
    "thirty days, and gift cards available in any amount. Sign up for our newsletter to get "
    "early access to new arrivals, exclusive coupons and seasonal style guides delivered to "
    "your inbox every week. Visit a store near you or shop online any time, day or night."
)


def page(title, body, nav=True):
    menu = "<nav>Home About Contact</nav>" if nav else ""
    return (f"<!doctype html><html><head><title>{title}</title><style>p{{}}</style>"
            f"<script>var x=1;</script></head><body>{menu}<p>{body}</p></body></html>")


ROUTES = {
    "/intact": (200, page("Mill site", ARTICLE)),
    "/edited": (200, page("Mill site", EDITED)),
    "/replaced": (200, page("Winter sale", OTHER)),
    "/gone": (404, page("Not here", "Nothing")),
    "/soft": (200, page("Page not found", "Sorry, the page you are looking for does not exist.")),
    "/parked": (200, page("example.test", "This domain is for sale! Make an offer on this domain today.")),
    "/blocked": (403, page("Forbidden", "No")),
    "/error": (503, page("Unavailable", "Try later")),
    "/challenge": (200, page("Just a moment...", "Checking your browser before accessing.")),
    "/": (200, page("Home", OTHER)),
    "/new-home-of-article": (200, page("Mill site", ARTICLE)),
    "/redesigned": (200, page("Mill site", OTHER + " " + ARTICLE + " " + OTHER.upper())),
}
REDIRECTS = {"/to-home": "/", "/moved": "/new-home-of-article"}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in REDIRECTS:
            self.send_response(301)
            self.send_header("Location", REDIRECTS[self.path])
            self.end_headers()
            return
        code, body = ROUTES.get(self.path, (404, "missing"))
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def fingerprint(text):
    return to_hex(simhash(text))


# ── fingerprints ────────────────────────────────────────────────────────────

def test_simhash_separates_same_edited_and_unrelated():
    _, extracted = html_to_text(page("t", ARTICLE))
    same = hamming(simhash(ARTICLE), simhash(extracted))
    edited = hamming(simhash(ARTICLE), simhash(EDITED))
    other = hamming(simhash(ARTICLE), simhash(OTHER))
    assert same <= 10 < edited or edited <= 22
    assert other > 22
    assert same < edited < other


def test_simhash_none_for_tiny_text():
    assert simhash("too short") is None


def test_html_to_text_skips_scripts_and_reads_title():
    title, text = html_to_text(page("Hello", "world words here"))
    assert title == "Hello"
    assert "var x" not in text and "world words here" in text


# ── sampler ─────────────────────────────────────────────────────────────────

def rows(n_per_crawl=50, crawls=("2015/10", "2020/05")):
    for crawl in crawls:
        for i in range(n_per_crawl):
            yield {"source": "common_crawl", "url": f"https://www.site{i}.com/page/{i}",
                   "crawl_partition": crawl, "created_dt": f"{crawl[:4]}-03-01T00:00:00Z",
                   "text": ARTICLE, "category": "web"}
    yield {"source": "reddit", "url": "https://reddit.com/r/x", "crawl_partition": "2015/10"}


def test_select_is_deterministic_capped_and_dedupes_to_earliest_crawl():
    a = select(rows(), per_crawl=10)
    b = select(rows(), per_crawl=10)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len({r["url"] for r in a}) == len(a)
    # identical URLs in both crawls: duplicates keep the earliest crawl
    assert all(r["crawl"] == "2015-10" for r in a)
    assert len(a) == 10
    assert all(r["host"].startswith("site") and r["tld"] == "com" for r in a)
    assert a[0]["simhash"] and a[0]["crawl_date"] == "2015-03-01"


def test_select_ignores_other_sources_and_bad_urls():
    out = select([{"source": "common_crawl", "url": "ftp://x", "crawl_partition": "2015/10"},
                  {"source": "reddit", "url": "https://a.com", "crawl_partition": "2015/10"}], 5)
    assert out == []


def test_normalize_url_and_last_words():
    assert normalize_url("HTTPS://Example.COM:443/a#frag") == "https://example.com/a"
    assert normalize_url("http://example.com") == "http://example.com/"
    assert normalize_url("javascript:alert(1)") is None
    words = last_words("A first line that is quite long " * 10)
    assert len(words) <= 90 and words.endswith("…")


# ── classifier (pure) ───────────────────────────────────────────────────────

def test_classify_errors():
    assert classify("http://a.com/x", None, Fetch(error="dns_nxdomain")).status == "domain_gone"
    assert classify("http://a.com/x", None, Fetch(error="timeout")).bucket == "unknown"


def test_classify_offsite_parking_host():
    f = Fetch(http_status=200, final_url="https://www.hugedomains.com/domain_profile.cfm?d=a.com",
              content_type="text/html", body=page("a.com", "Premium domain"))
    assert classify("http://a.com/x", fingerprint(ARTICLE), f).status == "parked"


def test_classify_unverifiable_content_is_reachable():
    f = Fetch(http_status=200, final_url="http://a.com/report.pdf", content_type="application/pdf")
    assert classify("http://a.com/report.pdf", fingerprint(ARTICLE), f).status == "reachable"


# ── checker end to end against a local server ───────────────────────────────

EXPECTED = {
    "/intact": "intact", "/edited": "edited", "/replaced": "replaced", "/gone": "not_found",
    "/soft": "soft_404", "/parked": "parked", "/blocked": "blocked", "/error": "server_error",
    "/challenge": "blocked", "/to-home": "redirect_home", "/moved": "moved",
}
SKETCH_EXPECTED = dict(EXPECTED, **{"/redesigned": "intact", "/": "reachable"})


def test_checker_end_to_end(server, tmp_path):
    records = [{"id": f"{i:016x}", "url": server + path, "host": "127.0.0.1",
                "crawl_date": "2015-03-01", "simhash": fingerprint(ARTICLE)}
               for i, path in enumerate(EXPECTED)]
    out = tmp_path / "results.jsonl"
    summary = checker.run(records, str(out), workers=4)
    assert summary["checked"] == len(EXPECTED)
    got = {r["id"]: r for r in checker.read_jsonl(str(out))}
    for i, (path, status) in enumerate(EXPECTED.items()):
        assert got[f"{i:016x}"]["status"] == status, (path, got[f"{i:016x}"])
    assert got[f"{list(EXPECTED).index('/moved'):016x}"]["redirects"] == 1

    # rerun resumes: nothing left to do
    again = checker.run(records, str(out), workers=4)
    assert again["checked"] == 0 and again["previously_done"] == len(EXPECTED)


def test_checker_with_sketch_handles_redesign_and_front_pages(server, tmp_path):
    from linkrot.fingerprint import sketch
    records = [{"id": f"{i:016x}", "url": server + path, "host": "127.0.0.1",
                "crawl_date": "2015-03-01", "simhash": fingerprint(ARTICLE), "sketch": sketch(ARTICLE)}
               for i, path in enumerate(SKETCH_EXPECTED)]
    out = tmp_path / "results.jsonl"
    checker.run(records, str(out), workers=4)
    got = {r["id"]: r for r in checker.read_jsonl(str(out))}
    for i, (path, status) in enumerate(SKETCH_EXPECTED.items()):
        assert got[f"{i:016x}"]["status"] == status, (path, got[f"{i:016x}"])
    intact = got[f"{list(SKETCH_EXPECTED).index('/intact'):016x}"]
    assert intact["retained"] == 1.0


def test_retained_estimates_share_of_wording():
    from linkrot.fingerprint import retained, sketch
    sk = sketch(ARTICLE)
    assert retained(sk, ARTICLE) == 1.0
    assert retained(sk, OTHER) < 0.1
    assert 0.35 <= retained(sk, EDITED) < 1.0
    assert retained(None, ARTICLE) is None and sketch("tiny") is None


def test_resume_survives_torn_tail(tmp_path):
    out = tmp_path / "r.jsonl"
    out.write_text(json.dumps({"id": "0" * 16, "status": "intact"}) + "\n{\"id\": \"tor")
    assert [r["id"] for r in checker.read_jsonl(str(out))] == ["0" * 16]


def test_error_kind_mapping():
    assert checker._error_kind(socket.gaierror(socket.EAI_NONAME, "x")) == "dns_nxdomain"
    assert checker._error_kind(socket.gaierror(socket.EAI_AGAIN, "x")) == "dns_temp"
    assert checker._error_kind(ConnectionRefusedError()) == "refused"
    assert checker._error_kind(TimeoutError()) == "timeout"


def test_shards_partition_ids():
    ids = [f"{i:016x}" for i in range(100)]
    parts = [[i for i in ids if checker.in_shard(i, s, 4)] for s in range(4)]
    assert sum(map(len, parts)) == 100 and not set(parts[0]) & set(parts[1])


# ── aggregation ─────────────────────────────────────────────────────────────

def synthetic_rows(n=6000, seed=1):
    """Fake census rows with a known decay rate, for aggregation tests and site previews."""
    import math
    import random
    from linkrot.classify import BUCKETS
    rng = random.Random(seed)
    lam = 0.07
    rows = []
    for i in range(n):
        year = rng.randint(2013, 2026)
        age = 2026.9 - (year + rng.random())
        r = rng.random()
        if r < 0.08:
            status = rng.choice(["blocked", "server_error", "unreachable"])
        elif rng.random() < math.exp(-lam * age):
            status = rng.choice(["intact"] * 5 + ["edited"] * 2 + ["moved", "reachable", "replaced"])
        else:
            status = rng.choice(["not_found"] * 4 + ["domain_gone"] * 3 +
                                ["soft_404", "redirect_home", "parked", "defunct"])
        host = f"site{rng.randint(1, 4000)}.{rng.choice(['com'] * 6 + ['org', 'net', 'uk', 'de', 'info'])}"
        rows.append({"id": f"{i:016x}", "url": f"https://{host}/p/{i}", "host": host,
                     "tld": host.rsplit('.', 1)[1], "crawl": f"{year}-10",
                     "crawl_date": f"{year}-{rng.randint(1, 12):02d}-15", "category": "web",
                     "last_words": f"Page {i} title words", "status": status,
                     "bucket": BUCKETS[status], "checked_at": "2026-11-21T12:00:00Z",
                     "distance": rng.randint(0, 40) if status in ("intact", "edited", "replaced") else None,
                     "archived": rng.random() < 0.6 if BUCKETS[status] == "dead" and rng.random() < 0.3 else None})
    return rows


def test_summarize_recovers_decay_and_bounds():
    from datetime import date
    from linkrot.aggregate import summarize, wilson
    s = summarize(synthetic_rows(), date(2026, 11, 21))
    assert s["sample"]["checked"] == 6000
    assert 0 < s["headline"]["dead_share"] < 1
    lam = s["decay"]["survival"]["lambda_per_year"]
    assert 0.05 < lam < 0.09  # true rate 0.07
    years = [y["year"] for y in s["by_year"]]
    assert years == sorted(years) and years[0] == 2013
    assert s["by_year"][0]["dead_share"] > s["by_year"][-1]["dead_share"]
    assert s["archive"]["dead_checked"] > 0
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    assert wilson(0, 0) == [None, None]


def test_graveyard_filters_unsafe_and_stratifies():
    from linkrot.aggregate import graveyard
    rows = synthetic_rows()
    rows[0].update(bucket="dead", status="not_found", url="https://casino-bonus.com/x")
    g = graveyard(rows, limit=140)
    assert g and all("casino" not in x["url"] for x in g)
    assert len({x["crawl_date"][:4] for x in g}) >= 10
    assert all(x["wayback"].startswith("https://web.archive.org/web/") for x in g)


def test_merge_later_results_win_and_history_dedupes(tmp_path):
    from linkrot.aggregate import merge, update_history
    sample = [{"id": "a", "url": "u"}]
    merged = merge(sample, [{"id": "a", "status": "unreachable"}, {"id": "a", "status": "intact"}],
                   [{"id": "a", "archived": True, "archive_ts": "2015"}])
    assert merged[0]["status"] == "intact" and merged[0]["archived"] is True
    path = tmp_path / "h.json"
    summary = {"census_date": "2026-11-21", "sample": {"checked": 1},
               "headline": {"dead_share": .1, "alive_share": .8, "drifted_share": .1, "unknown_share": 0}}
    path.write_text(json.dumps(update_history(str(path), summary)))
    assert len(update_history(str(path), summary)) == 1


def test_title_match_rescues_redesigned_articles_and_rejects_generic_titles():
    from linkrot.classify import title_match
    assert title_match("Clone Wars is about to reach Point of No Return", "Clone Wars is about to reach ‘Point of No Return’ – Club Jade")
    assert not title_match("Blog", "Blog - Blog")
    assert not title_match("Cheap flights from Khanty-Mansiysk to Ekaterinburg", "Cheap Flights, Airline Tickets & Airfares")


def test_classify_redirect_rules():
    from linkrot.fingerprint import sketch
    sk = sketch(ARTICLE)
    other = Fetch(http_status=200, content_type="text/html", body=page("Shop", OTHER))
    # deep link redirected to a section index
    f = Fetch(**{**other.__dict__, "final_url": "https://www.abt.com/blog"})
    assert classify("http://blog.abt.com/2011/10/some-post/", None, f, sk).status == "redirect_home"
    # deep link redirected to the front page with tracking parameters
    f = Fetch(**{**other.__dict__, "final_url": "https://www.aviasales.com/?market=ca"})
    assert classify("http://ca.jetradar.com/routes/hma/svx", None, f, sk).status == "redirect_home"
    # redirected to an error page
    f = Fetch(**{**other.__dict__, "final_url": "https://citywire.com/wealth-manager/pagenotfound?x=1"})
    assert classify("http://citywire.co.uk/wealth-manager/banks/a", None, f, sk).status == "soft_404"
    # same article, new design, little wording kept but the same title
    moved = Fetch(http_status=200, content_type="text/html", final_url="https://new.example/2012/11/mill-site-plans",
                  body=page("County planning board weighs three plans for the old mill site", OTHER + " " + ARTICLE[:180]))
    v = classify("http://old.example/2012/11/mill", None, moved, sk, "County planning board weighs three plans for the old mill site")
    assert v.status == "moved", v
