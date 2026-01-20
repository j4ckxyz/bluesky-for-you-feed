import os
import logging

from dotenv import load_dotenv

from server.logger import logger

load_dotenv()

SERVICE_DID = os.environ.get('SERVICE_DID')
HOSTNAME = os.environ.get('HOSTNAME')
FLASK_RUN_FROM_CLI = os.environ.get('FLASK_RUN_FROM_CLI')
SUBSCRIPTION_ENDPOINT = os.environ.get('SUBSCRIPTION_ENDPOINT', 'wss://bsky.network/xrpc')

if FLASK_RUN_FROM_CLI:
    logger.setLevel(logging.DEBUG)

if not HOSTNAME:
    raise RuntimeError('You should set "HOSTNAME" environment variable first.')

if not SERVICE_DID:
    SERVICE_DID = f'did:web:{HOSTNAME}'


FEED_URI = os.environ.get('FEED_URI')
if not FEED_URI:
    raise RuntimeError('Publish your feed first (run publish_feed.py) to obtain Feed URI. '
                       'Set this URI to "FEED_URI" environment variable.')


def _get_bool_env_var(value: str) -> bool:
    if value is None:
        return False

    normalized_value = value.strip().lower()
    if normalized_value in {'1', 'true', 't', 'yes', 'y'}:
        return True

    return False


def _get_int_env_var(value: str, default: int) -> int:
    if value is None:
        return default

    try:
        return int(value.strip())
    except ValueError:
        return default


def _get_float_env_var(value: str, default: float) -> float:
    if value is None:
        return default

    try:
        return float(value.strip())
    except ValueError:
        return default


IGNORE_ARCHIVED_POSTS = _get_bool_env_var(os.environ.get('IGNORE_ARCHIVED_POSTS'))
IGNORE_REPLY_POSTS = _get_bool_env_var(os.environ.get('IGNORE_REPLY_POSTS'))

ALLOWLIST_DIDS = {
    did.strip()
    for did in (os.environ.get('ALLOWLIST_DIDS') or '').split(',')
    if did.strip()
}

IN_NETWORK_MAX_AGE_HOURS = _get_int_env_var(os.environ.get('IN_NETWORK_MAX_AGE_HOURS'), 72)
OON_MAX_AGE_HOURS = _get_int_env_var(os.environ.get('OON_MAX_AGE_HOURS'), 48)
MAX_POST_AGE_HOURS = _get_int_env_var(os.environ.get('MAX_POST_AGE_HOURS'), 96)
SOCIAL_PROOF_MAX_AGE_HOURS = _get_int_env_var(os.environ.get('SOCIAL_PROOF_MAX_AGE_HOURS'), 48)
USER_ACTION_LOOKBACK_HOURS = _get_int_env_var(os.environ.get('USER_ACTION_LOOKBACK_HOURS'), 168)
SERVED_POST_TTL_HOURS = _get_int_env_var(os.environ.get('SERVED_POST_TTL_HOURS'), 36)

IN_NETWORK_CANDIDATE_LIMIT = _get_int_env_var(os.environ.get('IN_NETWORK_CANDIDATE_LIMIT'), 600)
OON_CANDIDATE_LIMIT = _get_int_env_var(os.environ.get('OON_CANDIDATE_LIMIT'), 600)
SOCIAL_PROOF_CANDIDATE_LIMIT = _get_int_env_var(os.environ.get('SOCIAL_PROOF_CANDIDATE_LIMIT'), 300)
MAX_CANDIDATES = _get_int_env_var(os.environ.get('MAX_CANDIDATES'), 1200)
USER_ACTION_HISTORY_LIMIT = _get_int_env_var(os.environ.get('USER_ACTION_HISTORY_LIMIT'), 200)

RECENCY_HALF_LIFE_HOURS = _get_float_env_var(os.environ.get('RECENCY_HALF_LIFE_HOURS'), 10.0)
WEIGHT_IN_NETWORK = _get_float_env_var(os.environ.get('WEIGHT_IN_NETWORK'), 1.4)
WEIGHT_OON = _get_float_env_var(os.environ.get('WEIGHT_OON'), 0.2)
WEIGHT_LIKE = _get_float_env_var(os.environ.get('WEIGHT_LIKE'), 1.0)
WEIGHT_REPOST = _get_float_env_var(os.environ.get('WEIGHT_REPOST'), 2.0)
WEIGHT_REPLY = _get_float_env_var(os.environ.get('WEIGHT_REPLY'), 1.5)
WEIGHT_AUTHOR_AFFINITY = _get_float_env_var(os.environ.get('WEIGHT_AUTHOR_AFFINITY'), 1.2)
WEIGHT_SOCIAL_PROOF = _get_float_env_var(os.environ.get('WEIGHT_SOCIAL_PROOF'), 0.9)
WEIGHT_MEDIA = _get_float_env_var(os.environ.get('WEIGHT_MEDIA'), 0.4)
WEIGHT_RECENCY = _get_float_env_var(os.environ.get('WEIGHT_RECENCY'), 2.0)
WEIGHT_REPLY_PENALTY = _get_float_env_var(os.environ.get('WEIGHT_REPLY_PENALTY'), -0.5)
AUTHOR_DIVERSITY_DECAY = _get_float_env_var(os.environ.get('AUTHOR_DIVERSITY_DECAY'), 0.6)
