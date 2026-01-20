# Bluesky For You Feed Generator Setup Guide

This guide walks through hosting, publishing, and tuning the Bluesky feed
generator in this repo. It assumes a VPS with a public IP and a DNS hostname.

## What you are deploying

- A feed generator service that subscribes to the global firehose
  (`wss://bsky.network/xrpc`).
- A custom ranking pipeline that mirrors the X For You stages using Bluesky
  signals (in-network, out-of-network, social proof, recency, diversity).
- An authenticated feed that uses the requesting user DID and supports an
  allowlist during testing.

## Requirements

- VPS with a public IPv4 or IPv6 address
- A DNS hostname pointing at that VPS (ex: `feed.example.com`)
- Python 3.9+
- HTTPS on port 443 (Caddy or Nginx)

## 1. Pick a hostname and set DNS

Choose a hostname for the feed generator (recommended: `feed.<your-domain>`).
Create A/AAAA records so `feed.<your-domain>` resolves to your VPS IP.

This hostname is used for:

- `HOSTNAME` in `.env`
- The feed generator DID: `did:web:<HOSTNAME>`
- The `/.well-known/did.json` endpoint served by the app

## 2. Clone the repo on the VPS

Once you push this repo to GitHub, you can clone it on the VPS:

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
```

## 3. Configure the environment

Copy the example env file and update values:

```bash
cp .env.example .env
```

Minimal values to start:

```text
HOSTNAME=feed.example.com
HANDLE=your.handle
PASSWORD=xxxx-xxxx-xxxx-xxxx
PDS_URL=https://pds.example.com
ALLOWLIST_DIDS=did:plc:exampledid
```

After publishing the feed (step 5), add:

```text
FEED_URI=at://did:plc:.../app.bsky.feed.generator/<record-name>
```

Notes:

- `PDS_URL` is required for custom PDS logins.
- Leave `ALLOWLIST_DIDS` empty to open the feed to all authenticated users.
- The feed subscribes to the global firehose by default. You can override it
  with `SUBSCRIPTION_ENDPOINT` if needed.

## 4. Install dependencies

```bash
./setupvenv.sh
```

This creates a virtualenv and installs dependencies from `requirements.txt`.

## 5. Publish the feed record (creates FEED_URI)

Publishing creates the feed record in your repo and prints the feed URI.

```bash
python publish_feed.py
```

Copy the printed URI into `.env` as `FEED_URI`.

If you want a new record name, set `RECORD_NAME` in `.env` before publishing.

## 6. Run the server

Development:

```bash
flask run
```

Production (recommended): run a WSGI server behind a TLS reverse proxy.

### Example Caddyfile

```text
feed.example.com {
  reverse_proxy 127.0.0.1:8080
}
```

### Example WSGI run command

```bash
waitress-serve --listen=127.0.0.1:8080 server.app:app
```

## 7. Systemd service (optional but recommended)

Create `/etc/systemd/system/bluesky-feed.service`:

```ini
[Unit]
Description=For You Feed Generator
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/<repo>
EnvironmentFile=/home/ubuntu/<repo>/.env
ExecStart=/home/ubuntu/<repo>/.venv/bin/waitress-serve --listen=127.0.0.1:8080 server.app:app
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable bluesky-feed.service
sudo systemctl start bluesky-feed.service
```

## 8. Verify the service

```bash
curl https://feed.example.com/.well-known/did.json
```

Then test a feed skeleton response (replace feed URI):

```bash
curl "https://feed.example.com/xrpc/app.bsky.feed.getFeedSkeleton?feed=<FEED_URI>&limit=5"
```

If you get 401 or 403, ensure:

- Your client is passing JWT auth.
- Your DID is in `ALLOWLIST_DIDS` (or the allowlist is empty).

## Tuning the feed

All weights and limits are in `.env.example` and read from `.env`. The most
impactful settings:

- `WEIGHT_IN_NETWORK` / `WEIGHT_OON`
- `WEIGHT_RECENCY`
- `WEIGHT_LIKE`, `WEIGHT_REPOST`, `WEIGHT_REPLY`
- `AUTHOR_DIVERSITY_DECAY`
- `IN_NETWORK_CANDIDATE_LIMIT`, `OON_CANDIDATE_LIMIT`

## Algorithm deep dive

### Candidate sources

- **In-network**: posts from followed accounts within `IN_NETWORK_MAX_AGE_HOURS`.
- **Out-of-network**: high-engagement posts from non-followed accounts within
  `OON_MAX_AGE_HOURS`.
- **Social proof**: posts liked/reposted by follows within
  `SOCIAL_PROOF_MAX_AGE_HOURS`.

### Filters

- Dedupe by URI
- Remove self posts, blocked authors, and expired posts
- Drop replies when `IGNORE_REPLY_POSTS=true`
- Remove recently served posts (session dedupe)

### Scoring

Scores combine engagement, social proof, author affinity, and recency:

```text
score = engagement
      + author_affinity
      + social_proof
      + base_in_network_or_oon
      + media_bonus
      + recency_bonus
      + reply_penalty
```

Engagement and social proof are log-scaled, recency uses an exponential decay
controlled by `RECENCY_HALF_LIFE_HOURS`.

### Diversity

After scoring, repeated authors are attenuated with `AUTHOR_DIVERSITY_DECAY`
to prevent back-to-back posts from the same account.

## Data model

- `Post`: indexed feed posts with engagement counters
- `Like`, `Repost`: engagement events and subjects
- `Follow`, `Block`: viewer graph data
- `UserAction`: actions for author affinity
- `ServedPost`: recently served posts for dedupe

## Optional ML / LLM enhancements

This implementation does not require an LLM. If you want richer relevance
signals, common options include:

- Embeddings for semantic retrieval:
  - OpenAI: `text-embedding-3-large`
  - Cohere: `embed-english-v3.0`
  - Open source: `bge-large-en`, `e5-large-v2`
- Lightweight content classification or topic filters:
  - OpenAI: `gpt-4o-mini`
  - Anthropic: `claude-3-haiku`
  - Open source: `llama-3-8b-instruct`

If you add embeddings, you can store vectors in Postgres/pgvector or a vector
DB and use them to power better out-of-network retrieval.

## Troubleshooting

- `did.json` 404: confirm `HOSTNAME` matches the domain and HTTPS is working.
- Feed empty: wait for firehose indexing, and confirm your allowlist is set.
- Old schema: delete `feed_database.db` after pulling changes to rebuild tables.
