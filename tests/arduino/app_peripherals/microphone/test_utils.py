# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import json
from unittest.mock import MagicMock, patch

import pytest

from arduino.app_peripherals.microphone.errors import MicrophoneOpenError
from arduino.app_peripherals.microphone.utils import _claim_first_available_microphone, list_audio_sources, _nth_plugged_microphone

_CARRIER_ENV = "CONFIGURED_CARRIERS"


@pytest.fixture(autouse=True)
def _no_carrier(monkeypatch):
    """Default to a non-media-carrier environment unless a test opts in."""
    monkeypatch.delenv(_CARRIER_ENV, raising=False)


class TestNthPluggedMicrophone:
    """External behavior of _nth_plugged_microphone device resolution."""

    def test_usb_first_without_carrier(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert _nth_plugged_microphone(0) == "usb:1"

    def test_out_of_range_without_carrier_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        # Second position has no USB device and there is no built-in fallback without media carrier.
        with pytest.raises(MicrophoneOpenError):
            _nth_plugged_microphone(1)

    def test_usb_precedence_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert _nth_plugged_microphone(0) == "usb:1"

    def test_jack_fallback_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        assert _nth_plugged_microphone(0) == "jack:1"

    def test_index_spans_usb_then_jack_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert _nth_plugged_microphone(1) == "jack:1"

    def test_second_jack_is_addressable_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52, 54))

        assert _nth_plugged_microphone(1) == "jack:2"

    def test_out_of_range_jack_index_raises_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(MicrophoneOpenError):
            _nth_plugged_microphone(1)

    def test_no_jack_fallback_without_carrier(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(MicrophoneOpenError):
            _nth_plugged_microphone(0)

    def test_no_jack_fallback_for_other_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "some-other-carrier")
        mock_pw_dump(usb_ids=(), builtin_ids=(52,))

        with pytest.raises(MicrophoneOpenError):
            _nth_plugged_microphone(0)

    def test_no_devices_raises(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(MicrophoneOpenError):
            _nth_plugged_microphone(0)


class TestClaimFirstAvailableMicrophone:
    """Claim-aware device resolution used by Microphone auto-selection."""

    def test_skips_already_claimed_microphones(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50, 60))

        assert _claim_first_available_microphone() == "plughw:CARD=SomeCard,DEV=0"
        assert _claim_first_available_microphone() == "plughw:CARD=AnotherCard,DEV=0"

    def test_falls_back_to_jack_under_media_carrier(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        assert _claim_first_available_microphone() == "plughw:CARD=SomeCard,DEV=0"
        assert _claim_first_available_microphone() == "pipewire:NODE=alsa_input.platform-sound.Source-52"

    def test_raises_when_all_microphones_are_claimed(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,))

        _claim_first_available_microphone()
        with pytest.raises(MicrophoneOpenError):
            _claim_first_available_microphone()

    def test_raises_when_no_microphone_is_plugged(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(), builtin_ids=())

        with pytest.raises(MicrophoneOpenError):
            _claim_first_available_microphone()

    def test_never_selects_bluetooth_or_hdmi_microphones(self, mock_pw_dump, monkeypatch):
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(builtin_ids=(54,), bluetooth_ids=(50,), hdmi_ids=(52,))

        assert _claim_first_available_microphone() == "pipewire:NODE=alsa_input.platform-sound.Source-54"


class TestListAudioSources:
    """Discovery contract: USB/built-in partitioning ordered by PipeWire node id."""

    def test_partitions_usb_and_builtin(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(52,))

        usb, builtin = list_audio_sources()

        assert [s["id"] for s in usb] == [50]
        assert [s["id"] for s in builtin] == [52]

    def test_excludes_bluetooth_sources(self, mock_pw_dump):
        mock_pw_dump(usb_ids=(50,), builtin_ids=(54,), bluetooth_ids=(52,))

        usb, builtin = list_audio_sources()

        assert [s["id"] for s in usb] == [50]
        assert [s["id"] for s in builtin] == [54]

    def test_excludes_hdmi_sources(self, mock_pw_dump):
        # The HDMI node has the lowest id: exclusion, not ordering, must keep it out.
        mock_pw_dump(builtin_ids=(54,), hdmi_ids=(50,))

        usb, builtin = list_audio_sources()

        assert usb == []
        assert [s["id"] for s in builtin] == [54]

    def test_orders_usb_by_ascending_node_id(self, mock_pw_dump):
        # Declared out of order; discovery must sort by node id (lowest first).
        mock_pw_dump(usb_ids=(60, 50), builtin_ids=())

        usb, _ = list_audio_sources()

        assert [s["id"] for s in usb] == [50, 60]

    def test_orders_builtin_by_alsa_path(self, mock_pw_dump):
        # Node ids are descending, but the ALSA path order (declaration order) must win.
        mock_pw_dump(builtin_ids=(54, 52))

        _, builtin = list_audio_sources()

        assert [s["id"] for s in builtin] == [54, 52]

    def test_classifies_non_usb_bus_as_builtin(self):
        # A source is USB only when its parent device reports device.bus == "usb".
        objects = [
            {"id": 100, "info": {"props": {"media.class": "Audio/Device", "device.bus-path": "platform-sound"}}},
            {
                "id": 50,
                "info": {
                    "props": {
                        "media.class": "Audio/Source",
                        "node.name": "alsa_input.platform-sound.Source-50",
                        "device.id": 100,
                    }
                },
            },
        ]
        with patch("arduino.app_peripherals.microphone.utils.subprocess.run") as run:
            run.return_value = MagicMock(stdout=json.dumps(objects))
            usb, builtin = list_audio_sources()

        assert usb == []
        assert [s["id"] for s in builtin] == [50]


class TestPipeWireUnavailable:
    """With pw-dump unusable, microphone discovery degrades to ALSA-only enumeration."""

    def test_low_level_discovery_still_reports_the_failure(self, mock_pw_dump):
        # list_audio_sources keeps its contract: the fallback lives above it.
        mock_pw_dump(unavailable=True)

        with pytest.raises(MicrophoneOpenError):
            list_audio_sources()

    def test_usb_cards_are_selected_by_index(self, mock_pw_dump, usb_cards):
        mock_pw_dump(unavailable=True)
        usb_cards(0, 1)

        assert _nth_plugged_microphone(0) == "usb:1"
        assert _nth_plugged_microphone(1) == "usb:2"

    def test_invalid_json_also_falls_back(self, usb_cards):
        usb_cards(0)

        with patch("arduino.app_peripherals.microphone.utils.subprocess.run") as run:
            run.return_value = MagicMock(stdout="not-json")
            assert _nth_plugged_microphone(0) == "usb:1"

    def test_non_usb_cards_are_offered_as_last_resort(self, mock_pw_dump, usb_cards):
        # No USB card at all: a built-in codec is better than no microphone.
        mock_pw_dump(unavailable=True)
        usb_cards()

        assert _nth_plugged_microphone(0) == "usb:1"
        assert _claim_first_available_microphone() == "plughw:CARD=SomeCard,DEV=0"

    def test_last_resort_skips_cards_that_are_not_microphones(self, mock_pw_dump, usb_cards):
        # Recording from an HDMI or loopback card would capture silence.
        mock_pw_dump(unavailable=True)
        usb_cards()

        with patch("alsaaudio.cards", return_value=["vc4hdmi0", "Loopback"]):
            with patch("alsaaudio.card_indexes", return_value=[0, 1]):
                with patch("alsaaudio.pcms", return_value=["plughw:CARD=vc4hdmi0,DEV=0", "plughw:CARD=Loopback,DEV=0"]):
                    with pytest.raises(MicrophoneOpenError):
                        _nth_plugged_microphone(0)

    def test_no_card_at_all_raises(self, mock_pw_dump, usb_cards):
        mock_pw_dump(unavailable=True)
        usb_cards()

        with patch("alsaaudio.cards", return_value=[]):
            with patch("alsaaudio.card_indexes", return_value=[]):
                with pytest.raises(MicrophoneOpenError):
                    _nth_plugged_microphone(0)

    def test_claim_skips_already_claimed_fallback_microphones(self, mock_pw_dump, usb_cards):
        mock_pw_dump(unavailable=True)
        usb_cards(0, 1)

        assert _claim_first_available_microphone() == "plughw:CARD=SomeCard,DEV=0"
        assert _claim_first_available_microphone() == "plughw:CARD=AnotherCard,DEV=0"

    def test_jack_is_not_offered_under_media_carrier(self, mock_pw_dump, usb_cards, monkeypatch):
        # Jack microphones are PipeWire nodes: without PipeWire they can't be opened.
        monkeypatch.setenv(_CARRIER_ENV, "media-carrier")
        mock_pw_dump(unavailable=True)
        usb_cards(0, 1)

        with pytest.raises(MicrophoneOpenError):
            _nth_plugged_microphone(2)
