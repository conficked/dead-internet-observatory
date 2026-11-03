"""
Steam Minion — scrapes public game reviews via the Steam Store API.
No API key required.
"""

import sys
from datetime import datetime, timezone

import requests

from minions.base_minion import BaseMinion


GAME_NAMES: dict[int, str] = {
    730:     "Counter-Strike 2",
    570:     "Dota 2",
    440:     "Team Fortress 2",
    1172470: "Apex Legends",
    1245620: "Elden Ring",
    292030:  "The Witcher 3",
    271590:  "GTA V",
    578080:  "PUBG",
    1091500: "Cyberpunk 2077",
    2767030: "Helldivers 2",
    881100:  "Baldurs Gate 3",
    2379780: "Palworld",
    220:     "Half-Life 2",
    400:     "Portal",
    620:     "Portal 2",
    4000:    "Garrys Mod",
    1517290: "Farming Simulator 22",
    1938090: "Call of Duty",
    1326470: "Sons of the Forest",
    1888160: "Starfield",
}

DEFAULT_APP_IDS = [
    730, 570, 440, 1172470, 1245620, 292030, 271590, 578080, 1091500, 2767030,
    881100, 2379780, 220, 400, 620, 4000,
]


class SteamBot(BaseMinion):

    REVIEWS_URL = "https://store.steampowered.com/appreviews/{app_id}"

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path=config_path, minion_name="steam")
        cfg = self.config.get("sources", {}).get("steam", {})
        self.app_ids: list[int] = cfg.get("app_ids", DEFAULT_APP_IDS)
        self.reviews_per_game: int = cfg.get("reviews_per_game", 100)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DeadInternetObservatory/1.0 (research)",
        })

    # ── Fetch helpers ──────────────────────────────────────────────────────────

    def _fetch_reviews(self, app_id: int) -> list[dict]:
        url = self.REVIEWS_URL.format(app_id=app_id)
        try:
            resp = self.session.get(url, params={
                "json":          1,
                "filter":        "recent",
                "language":      "english",
                "num_per_page":  min(self.reviews_per_game, 100),
                "review_type":   "all",
                "purchase_type": "all",
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            self.logger.warning(f"  App {app_id} request failed: {exc}")
            self.stats["errors"] += 1
            return []

        raw_reviews = data.get("reviews", [])
        self.stats["fetched"] += len(raw_reviews)

        records = []
        for r in raw_reviews:
            text = (r.get("review") or "").strip()
            if not text:
                self.stats["skipped"] += 1
                continue
            records.append({
                "app_id":           app_id,
                "game_name":        GAME_NAMES.get(app_id, str(app_id)),
                "review_id":        r.get("recommendationid", ""),
                "author_steam_id":  r.get("author", {}).get("steamid", ""),
                "text":             text,
                "language":         r.get("language", ""),
                "timestamp_created": r.get("timestamp_created", 0),
                "voted_up":         bool(r.get("voted_up", False)),
                "votes_up":         r.get("votes_up", 0),
                "votes_funny":      r.get("votes_funny", 0),
                "playtime_forever": r.get("author", {}).get("playtime_forever", 0),
                "received_for_free": bool(r.get("received_for_free", False)),
                "category":         "social",
            })
        return records

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self):
        if not self.config["sources"]["steam"].get("enabled", True):
            self.logger.info("Steam minion disabled in config — exiting")
            return
        self.logger.info(f"Starting — {len(self.app_ids)} games, "
                         f"{self.reviews_per_game} reviews each")
        all_records: list[dict] = []

        for app_id in self.app_ids:
            game = GAME_NAMES.get(app_id, str(app_id))
            reviews = self._fetch_reviews(app_id)
            self.logger.info(f"  {game} ({app_id}): {len(reviews)} reviews")
            all_records.extend(reviews)
            self.stats["processed"] += len(reviews)
            self.throttle(1.0)

        if all_records:
            self.save_bronze(all_records, source="steam")

        self.logger.info(f"Done — {len(all_records)} reviews saved")
        # A run that fetched nothing while every request errored is a broken source,
        # not a quiet day. Fail the job so it cannot report green while dead.
        if not all_records and self.stats["errors"]:
            raise SystemExit(
                f"Steam minion produced 0 reviews with {self.stats['errors']} request errors")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    bot = SteamBot(config_path=config_path)
    bot.run()
    bot.report_stats()
