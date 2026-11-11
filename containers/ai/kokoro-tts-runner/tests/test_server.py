# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""HTTP API of the runner, served on a loopback port with a scripted model."""

import http.client
import json
import threading

import pytest

from server import BadRequest, make_server, parse_request


@pytest.fixture
def api(synthesizer):
    server = make_server(synthesizer, 0, host="127.0.0.1")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
        try:
            payload = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
            connection.request(method, path, body=payload)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    yield request
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_health(api):
    status, _, body = api("GET", "/health")
    assert status == 200 and json.loads(body) == {"status": "ok"}


def test_voices_lists_each_voice_with_its_language(api):
    status, _, body = api("GET", "/voices")
    assert status == 200
    assert json.loads(body) == {
        "voices": [{"name": "af_heart", "language": "en-us"}, {"name": "if_sara", "language": "it"}],
        "default_voice": "af_heart",
        "sample_rate": 24000,
    }


def test_synthesize_streams_the_pcm_of_every_sentence(api):
    status, headers, body = api("POST", "/synthesize", {"text": "Hi. Hi.", "voice": "if_sara", "sample_rate": 48000})
    assert status == 200
    assert headers["Content-Type"] == "audio/pcm;rate=48000;channels=1"
    assert headers["Transfer-Encoding"] == "chunked"
    # Two sentences of 6 tokens with the pads, 10 ms each, 2 bytes per sample
    assert len(body) == 2 * 6 * 480 * 2


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"{", "not valid JSON"),
        ([], "must be a JSON object"),
        ({"text": "  "}, "'text' must be a non-empty string"),
        ({"text": "Hi.", "speed": 5}, "'speed' must be a number between 0.5 and 2.0"),
        ({"text": "Hi.", "sample_rate": 44100.5}, "'sample_rate' must be an integer"),
        ({"text": "Hi.", "voice": "jf_alpha"}, "Unknown voice 'jf_alpha'"),
    ],
)
def test_a_bad_request_is_answered_with_its_reason(api, body, message):
    status, _, response = api("POST", "/synthesize", body)
    assert status == 400
    assert message in json.loads(response)["error"]["message"]


def test_a_failure_of_the_model_is_a_500_not_an_empty_stream(api, session):
    session.fail = True
    status, _, body = api("POST", "/synthesize", {"text": "Hi."})
    assert status == 500
    assert "model exploded" in json.loads(body)["error"]["message"]


def test_unknown_paths_are_404(api):
    assert api("GET", "/nope")[0] == 404
    assert api("POST", "/voices", {})[0] == 404


def test_parse_request_fills_the_defaults():
    assert parse_request(b'{"text": "Hi."}') == {"text": "Hi.", "voice": "af_heart", "speed": 1.0, "sample_rate": 24000}
    with pytest.raises(BadRequest):
        parse_request(b'{"text": 3}')
