# Bluesky For You Feed Generator

This repo ships a Bluesky feed generator inspired by the X For You pipeline,
rebuilt with signals that exist in the Bluesky ecosystem. It subscribes to the
global firehose, builds in-network and out-of-network candidates, scores them
with engagement + recency + social proof, and serves a personalized feed.

## Features

- Global firehose indexing (works across all PDSs)
- In-network and out-of-network candidate sourcing
- Weighted ranking with recency, engagement, author affinity, and diversity
- Authenticated feed requests with an allowlist for safe rollout
- SQLite storage (easy to swap for Postgres)
- Optional LLM enrichment (topics, safety, embeddings)

## Quick start

```bash
cp .env.example .env
./setupvenv.sh
python publish_feed.py
```

Then copy the printed feed URI into `FEED_URI` in `.env` and run:

```bash
flask run
```

For production deployment, systemd, HTTPS, and tuning guidance, see `SETUP.md`.

## Repo layout

- `server/app.py`: Flask app + XRPC routes
- `server/data_filter.py`: firehose ingestion and indexing
- `server/algos/feed.py`: ranking pipeline and feed assembly
- `publish_feed.py`: publishes the feed record (supports custom PDS)
- `SETUP.md`: full deployment guide + tuning + ML/LLM options

## License

MIT
