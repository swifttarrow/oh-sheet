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


def test_upload_midi_rejects_unknown_extension(client):
    # Parity with the audio endpoint: filename extension must be .mid/.midi.
    response = client.post(
        "/v1/uploads/midi",
        files={"file": ("song.txt", b"MThd\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00",
                        "text/plain")},
    )
    assert response.status_code == 415


def test_upload_midi_rejects_arbitrary_bytes_with_midi_extension(client):
    # Security / integrity: /v1/uploads/midi must validate that the
    # uploaded bytes are actually a MIDI file, not just that the
    # filename happens to end in .mid. Without the magic-header check,
    # a client could upload ANY bytes (JSON, an executable, random
    # garbage) and get a successful 200 with a blob URI they could then
    # pass to /v1/jobs.
    #
    # The Standard MIDI File spec requires every SMF to begin with the
    # 4-byte "MThd" chunk header. We check only these 4 bytes — deeper
    # structural validation is the ingest stage's job.
    response = client.post(
        "/v1/uploads/midi",
        files={"file": ("totally_not_midi.mid", b"this is a plain text file",
                        "audio/midi")},
    )
    assert response.status_code == 415
    assert "midi" in response.json()["detail"].lower()


def test_upload_midi_rejects_empty_file(client):
    # Edge case of the same validation: zero-byte file can't start with MThd.
    response = client.post(
        "/v1/uploads/midi",
        files={"file": ("empty.mid", b"", "audio/midi")},
    )
    assert response.status_code == 415


def test_upload_audio_preserves_source_filename(client):
    # The refine stage falls back to source_filename for title inference
    # when no metadata is identifiable (see
    # docs/superpowers/specs/2026-04-13-refine-filename-fallback-design.md).
    # Round-tripping the filename through the upload response is the
    # contract the whole fallback path depends on.
    response = client.post(
        "/v1/uploads/audio",
        files={"file": ("My Heart Will Go On.mp3", b"ID3\x04fake mp3 bytes", "audio/mpeg")},
    )
    assert response.status_code == 200
    assert response.json()["source_filename"] == "My Heart Will Go On.mp3"


def test_upload_midi_preserves_source_filename(client):
    # Same contract as test_upload_audio_preserves_source_filename — the
    # refine stage consumes source_filename as a hint for LLM refinement
    # when other metadata is missing.
    smf_header = (
        b"MThd"
        b"\x00\x00\x00\x06"
        b"\x00\x00"
        b"\x00\x01"
        b"\x01\xe0"
    )
    response = client.post(
        "/v1/uploads/midi",
        files={"file": ("Chopin_Nocturne.mid", smf_header, "audio/midi")},
    )
    assert response.status_code == 200
    assert response.json()["source_filename"] == "Chopin_Nocturne.mid"


def test_upload_midi_accepts_valid_smf_header(client):
    # Regression guard: the fix must not reject real MIDI files. A
    # minimal valid SMF header starts with "MThd" + 4-byte length +
    # 6-byte header body (format, ntrks, division).
    smf_header = (
        b"MThd"              # magic
        b"\x00\x00\x00\x06"  # chunk length = 6
        b"\x00\x00"          # format = 0 (single track)
        b"\x00\x01"          # ntrks = 1
        b"\x01\xe0"          # division = 480 ticks/quarter
    )
    response = client.post(
        "/v1/uploads/midi",
        files={"file": ("valid.mid", smf_header, "audio/midi")},
    )
    assert response.status_code == 200
