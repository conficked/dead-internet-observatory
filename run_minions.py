#!/usr/bin/env python3
"""
CLI entry point — run one or all minions locally.

Usage:
    python run_minions.py all
    python run_minions.py reddit
    python run_minions.py news
    python run_minions.py wikipedia
    python run_minions.py hackernews
    python run_minions.py wayback
    python run_minions.py commoncrawl [--dry-run]
    python run_minions.py bluesky
    python run_minions.py fourchan
    python run_minions.py steam
    python run_minions.py youtube
    python run_minions.py linkedin
    python run_minions.py twitter
    python run_minions.py pipeline      # bronze → silver → gold
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def cmd_reddit(_args):
    from minions.reddit_bot import RedditMinion
    RedditMinion().run()


def cmd_news(_args):
    from minions.news_crawler_bot import NewsCrawlerMinion
    NewsCrawlerMinion().run()


def cmd_wikipedia(_args):
    from minions.wikipedia_bot import WikipediaMinion
    WikipediaMinion().run()


def cmd_commoncrawl(args):
    from minions.common_crawl_bot import CommonCrawlMinion
    CommonCrawlMinion().run(dry_run=getattr(args, "dry_run", False))


def cmd_hackernews(_args):
    from minions.hackernews_bot import HackerNewsMinion
    HackerNewsMinion().run()


def cmd_wayback(_args):
    from minions.wayback_bot import WaybackMinion
    WaybackMinion().run()


def cmd_bluesky(_args):
    from minions.bluesky_bot import BlueskyBot
    BlueskyBot().run()


def cmd_fourchan(_args):
    from minions.fourchan_bot import FourchanBot
    FourchanBot().run()


def cmd_steam(_args):
    from minions.steam_bot import SteamBot
    SteamBot().run()


def cmd_youtube(_args):
    from minions.youtube_bot import YoutubeBot
    YoutubeBot().run()


def cmd_linkedin(_args):
    from minions.linkedin_bot import LinkedinBot
    LinkedinBot().run()


def cmd_twitter(_args):
    from minions.twitter_bot import TwitterBot
    TwitterBot().run()


def cmd_pipeline(_args):
    from pipeline.bronze_ingestion import BronzeToSilverPipeline
    from pipeline.silver_processing import SilverToGoldPipeline
    BronzeToSilverPipeline().run_all()
    SilverToGoldPipeline().run()


def cmd_all(args):
    for fn in [cmd_reddit, cmd_news, cmd_wikipedia, cmd_hackernews,
               cmd_bluesky, cmd_fourchan, cmd_steam, cmd_linkedin, cmd_twitter,
               cmd_pipeline]:
        fn(args)


COMMANDS = {
    "all":         cmd_all,
    "reddit":      cmd_reddit,
    "news":        cmd_news,
    "wikipedia":   cmd_wikipedia,
    "hackernews":  cmd_hackernews,
    "wayback":     cmd_wayback,
    "commoncrawl": cmd_commoncrawl,
    "bluesky":     cmd_bluesky,
    "fourchan":    cmd_fourchan,
    "steam":       cmd_steam,
    "youtube":     cmd_youtube,
    "linkedin":    cmd_linkedin,
    "twitter":     cmd_twitter,
    "pipeline":    cmd_pipeline,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dead Internet Observatory — minion runner")
    parser.add_argument("command", choices=list(COMMANDS.keys()))
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse without writing (Common Crawl only)")
    args = parser.parse_args()
    COMMANDS[args.command](args)
