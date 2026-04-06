def test_upload_audio_returns_remote_audio_file(client):
    response = client.post(
        "/v1/uploads/audio",
        files={"file": ("song.mp3", b"ID3\x04fake mp3 bytes", "audio/mpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["format"] == "mp3"
    assert body["uri"].startswith("file://")
    assert body["content_hash"]


def test_upload_audio_rejects_unknown_format(client):
    response = client.post(
        "/v1/uploads/audio",
        files={"file": ("song.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_midi_returns_remote_midi_file(client):
    response = client.post(
        "/v1/uploads/midi",
        files={"file": ("song.mid", b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00", "audio/midi")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["uri"].startswith("file://")
    assert body["ticks_per_beat"] == 480
