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


def _get_str_env_var(value: str, default: str) -> str:
    if value is None:
        return default
    value = value.strip()
    return value if value else default


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

LLM_ENABLED = _get_bool_env_var(os.environ.get('LLM_ENABLED'))
LLM_PROVIDER = _get_str_env_var(os.environ.get('LLM_PROVIDER'), 'openrouter')
LLM_CLASSIFY_PROVIDER = _get_str_env_var(
    os.environ.get('LLM_CLASSIFY_PROVIDER'),
    LLM_PROVIDER,
)
LLM_EMBED_PROVIDER = _get_str_env_var(
    os.environ.get('LLM_EMBED_PROVIDER'),
    LLM_PROVIDER,
)
LLM_API_KEY = os.environ.get('LLM_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
MINIMAX_API_KEY = os.environ.get('MINIMAX_API_KEY')

LLM_BASE_URL = _get_str_env_var(os.environ.get('LLM_BASE_URL'), 'https://api.openai.com/v1')
OPENROUTER_BASE_URL = _get_str_env_var(
    os.environ.get('OPENROUTER_BASE_URL'),
    'https://openrouter.ai/api/v1',
)
GEMINI_BASE_URL = _get_str_env_var(
    os.environ.get('GEMINI_BASE_URL'),
    'https://generativelanguage.googleapis.com/v1beta',
)
MINIMAX_BASE_URL = _get_str_env_var(
    os.environ.get('MINIMAX_BASE_URL'),
    'https://api.minimax.chat/v1',
)
MINIMAX_CHAT_PATH = _get_str_env_var(
    os.environ.get('MINIMAX_CHAT_PATH'),
    '/text/chatcompletion',
)
MINIMAX_EMBED_PATH = _get_str_env_var(
    os.environ.get('MINIMAX_EMBED_PATH'),
    '/text/embeddings',
)
MINIMAX_GROUP_ID = os.environ.get('MINIMAX_GROUP_ID')

LLM_MODEL = _get_str_env_var(os.environ.get('LLM_MODEL'), 'gpt-4o-mini')
LLM_CLASSIFY_MODEL = os.environ.get('LLM_CLASSIFY_MODEL')
LLM_EMBEDDING_MODEL = _get_str_env_var(
    os.environ.get('LLM_EMBEDDING_MODEL'),
    'text-embedding-3-small',
)
OPENAI_MODEL = os.environ.get('OPENAI_MODEL')
OPENAI_EMBEDDING_MODEL = os.environ.get('OPENAI_EMBEDDING_MODEL')
OPENROUTER_MODEL = os.environ.get('OPENROUTER_MODEL')
OPENROUTER_EMBEDDING_MODEL = os.environ.get('OPENROUTER_EMBEDDING_MODEL')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL')
GEMINI_EMBEDDING_MODEL = os.environ.get('GEMINI_EMBEDDING_MODEL')
MINIMAX_MODEL = os.environ.get('MINIMAX_MODEL')
MINIMAX_EMBEDDING_MODEL = os.environ.get('MINIMAX_EMBEDDING_MODEL')

LLM_SAMPLE_RATE = _get_float_env_var(os.environ.get('LLM_SAMPLE_RATE'), 0.02)
LLM_MIN_LIKES = _get_int_env_var(os.environ.get('LLM_MIN_LIKES'), 5)
LLM_MIN_REPOSTS = _get_int_env_var(os.environ.get('LLM_MIN_REPOSTS'), 2)
LLM_MIN_REPLIES = _get_int_env_var(os.environ.get('LLM_MIN_REPLIES'), 2)
LLM_MAX_CALLS_PER_MINUTE = _get_int_env_var(os.environ.get('LLM_MAX_CALLS_PER_MINUTE'), 10)
LLM_MAX_QUEUE = _get_int_env_var(os.environ.get('LLM_MAX_QUEUE'), 500)
LLM_MAX_TEXT_CHARS = _get_int_env_var(os.environ.get('LLM_MAX_TEXT_CHARS'), 1200)

LLM_TOPIC_BOOST = _get_float_env_var(os.environ.get('LLM_TOPIC_BOOST'), 0.6)
LLM_EMBEDDING_BOOST = _get_float_env_var(os.environ.get('LLM_EMBEDDING_BOOST'), 0.8)
LLM_SAFETY_PENALTY = _get_float_env_var(os.environ.get('LLM_SAFETY_PENALTY'), -1.2)
LLM_SAFETY_BLOCKLIST = {
    label.strip()
    for label in (os.environ.get('LLM_SAFETY_BLOCKLIST') or 'graphic').split(',')
    if label.strip()
}
LLM_SAFETY_PENALTY_LABELS = {
    label.strip()
    for label in (os.environ.get('LLM_SAFETY_PENALTY_LABELS') or 'nsfw,suggestive').split(',')
    if label.strip()
}

GRAPH_FOLLOWS_REFRESH_HOURS = _get_int_env_var(
    os.environ.get('GRAPH_FOLLOWS_REFRESH_HOURS'),
    24,
)
GRAPH_BLOCKS_REFRESH_HOURS = _get_int_env_var(
    os.environ.get('GRAPH_BLOCKS_REFRESH_HOURS'),
    24,
)
GRAPH_LIKES_REFRESH_HOURS = _get_int_env_var(
    os.environ.get('GRAPH_LIKES_REFRESH_HOURS'),
    6,
)
PDS_CACHE_REFRESH_HOURS = _get_int_env_var(
    os.environ.get('PDS_CACHE_REFRESH_HOURS'),
    24,
)
GRAPH_MAX_FOLLOWS = _get_int_env_var(os.environ.get('GRAPH_MAX_FOLLOWS'), 20000)
GRAPH_MAX_BLOCKS = _get_int_env_var(os.environ.get('GRAPH_MAX_BLOCKS'), 5000)
GRAPH_MAX_LIKES = _get_int_env_var(os.environ.get('GRAPH_MAX_LIKES'), 500)

TOPIC_PREF_MORE_DELTA = _get_float_env_var(os.environ.get('TOPIC_PREF_MORE_DELTA'), 1.0)
TOPIC_PREF_LESS_DELTA = _get_float_env_var(os.environ.get('TOPIC_PREF_LESS_DELTA'), -1.0)
TOPIC_PREF_MIN = _get_float_env_var(os.environ.get('TOPIC_PREF_MIN'), -4.0)
TOPIC_PREF_MAX = _get_float_env_var(os.environ.get('TOPIC_PREF_MAX'), 4.0)
TOPIC_PREF_BOOST = _get_float_env_var(os.environ.get('TOPIC_PREF_BOOST'), 1.0)

EMBED_TOPIC_SEEDS = [
    topic.strip()
    for topic in (
        os.environ.get(
            'EMBED_TOPIC_SEEDS',
            'art,photography,music,gaming,science,technology,sports,books,film,food,travel,nature,politics'
        )
    ).split(',')
    if topic.strip()
]
TOPIC_MIN_SCORE = _get_float_env_var(os.environ.get('TOPIC_MIN_SCORE'), 0.22)
TOPIC_MAX_COUNT = _get_int_env_var(os.environ.get('TOPIC_MAX_COUNT'), 4)
TOPIC_BLOCKLIST = {
    topic.strip().lower()
    for topic in (os.environ.get('TOPIC_BLOCKLIST') or 'politics').split(',')
    if topic.strip()
}

FOLLOW_NETWORK_WINDOW_HOURS = _get_int_env_var(os.environ.get('FOLLOW_NETWORK_WINDOW_HOURS'), 72)
FOLLOW_NETWORK_MIN_LIKES = _get_int_env_var(os.environ.get('FOLLOW_NETWORK_MIN_LIKES'), 2)
FOLLOW_NETWORK_MIN_REPOSTS = _get_int_env_var(os.environ.get('FOLLOW_NETWORK_MIN_REPOSTS'), 1)
FOLLOW_NETWORK_CANDIDATE_LIMIT = _get_int_env_var(
    os.environ.get('FOLLOW_NETWORK_CANDIDATE_LIMIT'),
    800,
)
FOLLOW_NETWORK_LIKE_WEIGHT = _get_float_env_var(os.environ.get('FOLLOW_NETWORK_LIKE_WEIGHT'), 2.0)
FOLLOW_NETWORK_REPOST_WEIGHT = _get_float_env_var(os.environ.get('FOLLOW_NETWORK_REPOST_WEIGHT'), 2.4)

SIMILARITY_MAX_AGE_HOURS = _get_int_env_var(os.environ.get('SIMILARITY_MAX_AGE_HOURS'), 48)
SIMILARITY_CANDIDATE_LIMIT = _get_int_env_var(os.environ.get('SIMILARITY_CANDIDATE_LIMIT'), 600)
EMBEDDING_SIMILARITY_MIN = _get_float_env_var(os.environ.get('EMBEDDING_SIMILARITY_MIN'), 0.18)
EMBEDDING_PROFILE_SIZE = _get_int_env_var(os.environ.get('EMBEDDING_PROFILE_SIZE'), 20)
