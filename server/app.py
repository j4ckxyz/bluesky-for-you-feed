import sys
import signal
import threading

from server import config
from server import data_stream

from flask import Flask, jsonify, request

from server.algos import algos
from server.data_filter import operations_callback
from server.llm_enricher import llm_enricher
from server.user_context import apply_interactions, get_viewer_context

app = Flask(__name__)

stream_stop_event = threading.Event()
stream_thread = threading.Thread(
    target=data_stream.run,
    args=(
        config.SERVICE_DID,
        operations_callback,
        stream_stop_event,
        config.SUBSCRIPTION_ENDPOINT,
    )
)
stream_thread.start()
llm_enricher.start()


def sigint_handler(*_):
    print('Stopping data stream...')
    stream_stop_event.set()
    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)


@app.route('/')
def index():
    return 'Bluesky For You Feed Generator'


@app.route('/.well-known/did.json', methods=['GET'])
def did_json():
    if not config.SERVICE_DID.endswith(config.HOSTNAME):
        return '', 404

    return jsonify({
        '@context': ['https://www.w3.org/ns/did/v1'],
        'id': config.SERVICE_DID,
        'service': [
            {
                'id': '#bsky_fg',
                'type': 'BskyFeedGenerator',
                'serviceEndpoint': f'https://{config.HOSTNAME}'
            }
        ]
    })


@app.route('/xrpc/app.bsky.feed.describeFeedGenerator', methods=['GET'])
def describe_feed_generator():
    feeds = [{'uri': uri} for uri in algos.keys()]
    response = {
        'encoding': 'application/json',
        'body': {
            'did': config.SERVICE_DID,
            'feeds': feeds
        }
    }
    return jsonify(response)


@app.route('/xrpc/app.bsky.feed.getFeedSkeleton', methods=['GET'])
def get_feed_skeleton():
    feed = request.args.get('feed', default=None, type=str)
    algo = algos.get(feed)
    if not algo:
        return 'Unsupported algorithm', 400

    from server.auth import AuthorizationError, get_auth_token, validate_auth
    try:
        requester_did = validate_auth(request)
        auth_token = get_auth_token(request)
    except AuthorizationError:
        return 'Unauthorized', 401

    if config.ALLOWLIST_DIDS and requester_did not in config.ALLOWLIST_DIDS:
        return 'Feed not available for this user yet', 403

    try:
        cursor = request.args.get('cursor', default=None, type=str)
        limit = request.args.get('limit', default=20, type=int)
        viewer_context = get_viewer_context(requester_did, auth_token)
        body = algo(cursor, limit, requester_did, viewer_context)
    except ValueError:
        return 'Malformed cursor', 400

    return jsonify(body)


@app.route('/xrpc/app.bsky.feed.sendInteractions', methods=['POST'])
def send_interactions():
    from server.auth import AuthorizationError, get_auth_token, validate_auth
    try:
        requester_did = validate_auth(request)
        _ = get_auth_token(request)
    except AuthorizationError:
        return 'Unauthorized', 401

    payload = request.get_json(silent=True) or {}
    interactions = payload.get('interactions') or []
    if not isinstance(interactions, list):
        return 'Invalid interactions payload', 400

    apply_interactions(requester_did, interactions)
    return jsonify({})
