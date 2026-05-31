#!/usr/bin/env python3
# coding=utf-8
"""
Generic DWT-DCT-SVD watermark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from pywt import dwt2, idwt2


BLOCK_SHAPE = (4, 4)


def text_to_bits(text: str) -> np.ndarray:
    data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    return np.unpackbits(data).astype(np.uint8)


def bits_to_text(bits: np.ndarray) -> str:
    bit_array = np.asarray(bits, dtype=np.uint8)
    if bit_array.size % 8:
        pad = 8 - bit_array.size % 8
        bit_array = np.concatenate([bit_array, np.zeros(pad, dtype=np.uint8)])
    data = np.packbits(bit_array).tobytes()
    return data.decode("utf-8", errors="replace")


def save_bgr(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(img, 0, 255).astype(np.uint8))


def block_view(ca: np.ndarray, block_shape: tuple[int, int] = BLOCK_SHAPE) -> tuple[np.ndarray, tuple[int, int]]:
    block_h, block_w = block_shape
    rows = ca.shape[0] // block_h
    cols = ca.shape[1] // block_w
    cropped = ca[: rows * block_h, : cols * block_w]
    blocks = (
        cropped.reshape(rows, block_h, cols, block_w)
        .swapaxes(1, 2)
        .reshape(rows * cols, block_h, block_w)
    )
    return blocks, (rows, cols)


def blocks_to_ca(blocks: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    rows, cols = grid_shape
    block_h, block_w = blocks.shape[1:]
    return blocks.reshape(rows, cols, block_h, block_w).swapaxes(1, 2).reshape(rows * block_h, cols * block_w)


def modulate_singular_values(s: np.ndarray, bit: int, d1: float, d2: float) -> np.ndarray:
    values = s.copy()
    values[0] = (values[0] // d1 + 1 / 4 + 1 / 2 * bit) * d1
    if d2 and len(values) > 1:
        values[1] = (values[1] // d2 + 1 / 4 + 1 / 2 * bit) * d2
    return values


class SimpleWatermarkDemo:
    def __init__(self, d1: float = 36, d2: float = 20, block_shape: tuple[int, int] = BLOCK_SHAPE):
        self.d1 = d1
        self.d2 = d2
        self.block_shape = block_shape

    def embed(self, img_bgr: np.ndarray, wm_bits: np.ndarray) -> np.ndarray:
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (height, width, 3).")
        if wm_bits.size == 0:
            raise ValueError("Watermark bit array must not be empty.")

        img_float = img_bgr.astype(np.float32)
        yuv = cv2.cvtColor(img_float, cv2.COLOR_BGR2YUV)

        embedded_channels = []
        for channel in range(3):
            ca, hvd = dwt2(yuv[:, :, channel], "haar")
            blocks, grid_shape = block_view(ca, self.block_shape)
            if wm_bits.size >= len(blocks):
                raise ValueError(f"Watermark has {wm_bits.size} bits, but only {len(blocks)} blocks are available.")

            new_blocks = blocks.copy()
            for idx, block in enumerate(blocks):
                bit = int(wm_bits[idx % wm_bits.size])
                block_dct = cv2.dct(block.astype(np.float32))
                u, s, vh = np.linalg.svd(block_dct)
                s_new = modulate_singular_values(s, bit, self.d1, self.d2)
                new_dct = u @ np.diag(s_new) @ vh
                new_blocks[idx] = cv2.idct(new_dct.astype(np.float32))

            ca_embedded = ca.copy()
            ca_part = blocks_to_ca(new_blocks, grid_shape)
            ca_embedded[: ca_part.shape[0], : ca_part.shape[1]] = ca_part
            embedded_channel = idwt2((ca_embedded, hvd), "haar")
            embedded_channels.append(embedded_channel[: img_bgr.shape[0], : img_bgr.shape[1]])

        embedded_yuv = np.stack(embedded_channels, axis=2)
        embedded_bgr = cv2.cvtColor(embedded_yuv, cv2.COLOR_YUV2BGR)
        return np.clip(embedded_bgr, 0, 255).astype(np.uint8)

    def extract_bits(self, embedded_bgr: np.ndarray, wm_size: int) -> np.ndarray:
        yuv = cv2.cvtColor(embedded_bgr.astype(np.float32), cv2.COLOR_BGR2YUV)
        block_votes = []
        for channel in range(3):
            ca, _ = dwt2(yuv[:, :, channel], "haar")
            blocks, _ = block_view(ca, self.block_shape)
            votes = np.zeros(len(blocks), dtype=np.float32)
            for idx, block in enumerate(blocks):
                block_dct = cv2.dct(block.astype(np.float32))
                _, s, _ = np.linalg.svd(block_dct)
                bit0 = float(s[0] % self.d1 > self.d1 / 2)
                if self.d2 and len(s) > 1:
                    bit1 = float(s[1] % self.d2 > self.d2 / 2)
                    votes[idx] = (3 * bit0 + bit1) / 4
                else:
                    votes[idx] = bit0
            block_votes.append(votes)

        vote_matrix = np.vstack(block_votes)
        extracted = np.zeros(wm_size, dtype=np.uint8)
        for bit_idx in range(wm_size):
            extracted[bit_idx] = int(vote_matrix[:, bit_idx::wm_size].mean() >= 0.5)
        return extracted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic DWT-DCT-SVD watermark demo.")
    parser.add_argument("--input", type=Path, default=Path("./input.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--watermark", default="SignalSystem")
    parser.add_argument("--d1", type=float, default=40)
    parser.add_argument("--d2", type=float, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wm_bits = text_to_bits(args.watermark)
    demo = SimpleWatermarkDemo(d1=args.d1, d2=args.d2)
    embedded = demo.embed(img, wm_bits)
    extracted_bits = demo.extract_bits(embedded, wm_size=wm_bits.size)
    extracted_text = bits_to_text(extracted_bits)

    save_bgr(output_dir / "watermarked.png", embedded)
    with (output_dir / "result.txt").open("w", encoding="utf-8") as f:
        f.write(f"watermark_text={args.watermark}\n")
        f.write(f"watermark_bits={wm_bits.size}\n")
        f.write(f"extracted_text={extracted_text}\n")
        f.write(f"bit_errors={int(np.count_nonzero(wm_bits != extracted_bits))}\n")

    print(f"Watermark text: {args.watermark}")
    print(f"Extracted text: {extracted_text}")
    print(f"Bit errors: {int(np.count_nonzero(wm_bits != extracted_bits))}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
