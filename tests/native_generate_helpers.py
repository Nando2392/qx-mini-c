"""Private QXT fixtures for native-generation acceptance tests.

These helpers copy or construct tokenizer sidecars only.  They never modify the
checked-in real tokenizer or model and do not synthesize model outputs.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Mapping, Sequence


QXT_HEADER = struct.Struct("<4sIIIIIiiIIQ")


def fnv1a64(data: bytes) -> int:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def copy_qxt_with_eos(source: Path, destination: Path, eos_token_id: int) -> Path:
    """Copy a real QXT and alter only its EOS header metadata."""
    raw = bytearray(source.read_bytes())
    header = QXT_HEADER.unpack_from(raw)
    assert header[0] == b"QXT2" and header[1] == 2
    assert -1 <= eos_token_id < header[2]
    struct.pack_into("<i", raw, 28, eos_token_id)
    destination.write_bytes(raw)
    return destination


def copy_qxt_with_token_pieces(
    source: Path,
    destination: Path,
    replacements: Mapping[int, str],
    *,
    eos_token_id: int = -1,
) -> Path:
    """Copy a QXT while replacing selected vocabulary pieces.

    The complete token table is rebuilt and the payload checksum is recomputed.
    This is a controlled test fixture: inference remains the real model path.
    """
    raw = source.read_bytes()
    header = list(QXT_HEADER.unpack_from(raw))
    magic, version, vocab_count, merge_count, model_len, pre_len = header[:6]
    assert magic == b"QXT2" and version == 2
    assert all(0 <= token_id < vocab_count for token_id in replacements)
    assert -1 <= eos_token_id < vocab_count

    payload = raw[QXT_HEADER.size :]
    cursor = model_len + pre_len
    prefix = payload[:cursor]
    records: list[tuple[bytes, int]] = []
    for token_id in range(vocab_count):
        length, token_type = struct.unpack_from("<II", payload, cursor)
        cursor += 8
        piece = payload[cursor : cursor + length]
        cursor += length
        records.append((piece, token_type))

    replacement_bytes = {token_id: piece.encode("utf-8") for token_id, piece in replacements.items()}
    assert len(set(replacement_bytes.values())) == len(replacement_bytes)
    owner_by_piece = {piece: token_id for token_id, (piece, _) in enumerate(records)}
    assert len(owner_by_piece) == vocab_count

    # Real GPT-2/Qwen vocabularies already contain the byte-unicode pieces used
    # by split-UTF-8 tests.  Swap each requested piece with its existing owner
    # instead of introducing a duplicate, which the native encoder rejects.
    for token_id, piece in replacement_bytes.items():
        owner = owner_by_piece[piece]
        assert owner not in replacements
        old_piece, token_type = records[token_id]
        owner_piece, owner_type = records[owner]
        assert owner_piece == piece
        records[token_id] = (piece, token_type)
        records[owner] = (old_piece, owner_type)

    rebuilt = bytearray(prefix)
    for piece, token_type in records:
        rebuilt += struct.pack("<II", len(piece), token_type)
        rebuilt += piece
    rebuilt += payload[cursor:]  # Preserve the real merge table byte-for-byte.

    header[7] = eos_token_id
    header[9] = len(rebuilt)
    header[10] = fnv1a64(rebuilt)
    destination.write_bytes(QXT_HEADER.pack(*header) + rebuilt)
    return destination


def write_mini_qxt(
    destination: Path,
    tokens: Sequence[str],
    *,
    eos_token_id: int = -1,
) -> Path:
    """Write a deliberately miniature QXT tokenizer-only fixture."""
    assert tokens
    assert -1 <= eos_token_id < len(tokens)
    model = b"gpt2"
    pre = b"qwen2"
    payload = bytearray(model + pre)
    for token in tokens:
        piece = token.encode("utf-8")
        payload += struct.pack("<II", len(piece), 1)
        payload += piece
    header = QXT_HEADER.pack(
        b"QXT2",
        2,
        len(tokens),
        0,
        len(model),
        len(pre),
        -1,
        eos_token_id,
        0,
        len(payload),
        fnv1a64(payload),
    )
    destination.write_bytes(header + payload)
    return destination
