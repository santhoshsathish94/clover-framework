"""Android native media helpers (Chaquopy -> Android OS MediaMetadataRetriever, MediaExtractor & MediaCodec).

Lets LFM3-Omni-2B on Android phones (where a standalone ffmpeg binary is not shipped):
1. Transcode attached .m4a / .ogg / .webm / .aac voice notes (e.g. Samsung Voice Recorder) into 16 kHz mono WAV.
2. Extract 2 fps multi-frame keyframes AND the 16 kHz mono WAV soundtrack from attached .mp4 / .webm videos so
   LFM3-Omni-2B sees the video frames and hears its soundtrack together on Android just as on Windows.
"""
from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image

import paths


def _pcm16_mono_wav(samples_i16: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = np.asarray(samples_i16, dtype="<i2").tobytes()
    hdr = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        int(sample_rate),
        int(sample_rate) * 2,
        2,
        16,
        b"data",
        len(pcm),
    )
    return hdr + pcm


def decode_audio_to_wav16k(src_path: str | Path, dst_wav_path: str | Path, max_seconds: float = 180.0) -> bool:
    """Decode an audio file (or the audio track of a video file) into a 16 kHz mono PCM16 WAV using Android MediaCodec."""
    if not paths.ANDROID:
        return False
    try:
        from java import jclass
    except Exception:
        return False

    MediaExtractor = jclass("android.media.MediaExtractor")
    MediaCodec = jclass("android.media.MediaCodec")
    MediaFormat = jclass("android.media.MediaFormat")
    BufferInfo = jclass("android.media.MediaCodec$BufferInfo")

    extractor = MediaExtractor()
    codec = None
    try:
        extractor.setDataSource(str(src_path))
        n_tracks = extractor.getTrackCount()
        audio_track = -1
        fmt = None
        mime = ""
        for i in range(n_tracks):
            tf = extractor.getTrackFormat(i)
            m = str(tf.getString(MediaFormat.KEY_MIME) or "")
            if m.startswith("audio/"):
                audio_track = i
                fmt = tf
                mime = m
                break
        if audio_track < 0 or fmt is None:
            return False

        extractor.selectTrack(audio_track)
        in_rate = int(fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)) if fmt.containsKey(MediaFormat.KEY_SAMPLE_RATE) else 44100
        channels = int(fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)) if fmt.containsKey(MediaFormat.KEY_CHANNEL_COUNT) else 1
        channels = max(1, channels)

        codec = MediaCodec.createDecoderByType(mime)
        codec.configure(fmt, None, None, 0)
        codec.start()

        info = BufferInfo()
        saw_in_eos = False
        saw_out_eos = False
        chunks: list[np.ndarray] = []
        total_frames = 0
        max_in_frames = int(max_seconds * max(8000, in_rate))

        while not saw_out_eos and total_frames < max_in_frames:
            if not saw_in_eos:
                in_idx = codec.dequeueInputBuffer(10000)
                if in_idx >= 0:
                    in_buf = codec.getInputBuffer(in_idx)
                    sample_sz = extractor.readSampleData(in_buf, 0)
                    if sample_sz < 0:
                        codec.queueInputBuffer(in_idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        saw_in_eos = True
                    else:
                        pts = extractor.getSampleTime()
                        codec.queueInputBuffer(in_idx, 0, sample_sz, max(0, pts), 0)
                        extractor.advance()

            out_idx = codec.dequeueOutputBuffer(info, 10000)
            if out_idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED:
                ofmt = codec.getOutputFormat()
                if ofmt.containsKey(MediaFormat.KEY_SAMPLE_RATE):
                    in_rate = int(ofmt.getInteger(MediaFormat.KEY_SAMPLE_RATE))
                if ofmt.containsKey(MediaFormat.KEY_CHANNEL_COUNT):
                    channels = max(1, int(ofmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)))
            elif out_idx >= 0:
                if info.size > 0:
                    out_buf = codec.getOutputBuffer(out_idx)
                    out_buf.position(info.offset)
                    out_buf.limit(info.offset + info.size)
                    raw_b = bytes(bytearray(info.size))
                    # Copy bytes from Java ByteBuffer
                    arr_j = bytearray(info.size)
                    for k in range(info.size):
                        arr_j[k] = out_buf.get() & 0xFF
                    raw_b = bytes(arr_j)
                    s16 = np.frombuffer(raw_b[: (len(raw_b) // 2) * 2], dtype="<i2")
                    if channels > 1 and s16.size >= channels:
                        s16 = s16[: (s16.size // channels) * channels].reshape(-1, channels).mean(axis=1).astype(np.int16)
                    if s16.size > 0:
                        chunks.append(s16)
                        total_frames += int(s16.size)
                codec.releaseOutputBuffer(out_idx, False)
                if (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0:
                    saw_out_eos = True

        if not chunks:
            return False
        mono = np.concatenate(chunks)
        if in_rate != 16000 and mono.size > 1:
            out_len = max(1, int(round(mono.size * 16000.0 / float(in_rate))))
            x_old = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=False)
            mono = np.clip(np.round(np.interp(x_new, x_old, mono.astype(np.float32))), -32768, 32767).astype("<i2")

        Path(dst_wav_path).write_bytes(_pcm16_mono_wav(mono, 16000))
        return Path(dst_wav_path).is_file() and Path(dst_wav_path).stat().st_size > 44
    except Exception:
        return False
    finally:
        if codec is not None:
            try:
                codec.stop()
            except Exception:
                pass
            try:
                codec.release()
            except Exception:
                pass
        try:
            extractor.release()
        except Exception:
            pass


def extract_video_frames_and_audio(
    video_path: str | Path,
    media_dir: str | Path,
    *,
    fps: float = 2.0,
    min_frames: int = 4,
    max_frames: int = 16,
    max_edge: int = 672,
) -> tuple[list[str], str | None]:
    """Extract multi-frame PNGs (at ~2 fps, 4..16 frames) and optional 16 kHz WAV soundtrack from a video on Android."""
    if not paths.ANDROID:
        return [], None
    video_p = Path(video_path)
    media_p = Path(media_dir)
    if not video_p.is_file():
        return [], None
    stem = video_p.stem[:24]
    cached_meta = media_p / f"vidparts_{stem}.txt"
    if cached_meta.is_file():
        try:
            lines = [ln.strip() for ln in cached_meta.read_text(encoding="utf-8").splitlines() if ln.strip()]
            frames = [ln[2:] for ln in lines if ln.startswith("F:") and (media_p / ln[2:]).is_file()]
            aud = next((ln[2:] for ln in lines if ln.startswith("A:") and (media_p / ln[2:]).is_file()), None)
            if frames:
                return frames, aud
        except Exception:
            pass

    frames_out: list[str] = []
    try:
        from java import jclass
        MediaMetadataRetriever = jclass("android.media.MediaMetadataRetriever")
        BitmapCompressFormat = jclass("android.graphics.Bitmap$CompressFormat")
        ByteArrayOutputStream = jclass("java.io.ByteArrayOutputStream")

        mmr = MediaMetadataRetriever()
        try:
            mmr.setDataSource(str(video_p))
            dur_str = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            dur_ms = int(dur_str) if (dur_str and str(dur_str).isdigit()) else 2000
            dur_s = max(0.5, dur_ms / 1000.0)
            n_frames = max(min_frames, min(max_frames, int(round(dur_s * fps))))
            times_us = [int((dur_ms * 1000 * i) / max(1, n_frames)) for i in range(n_frames)]
            seen_hashes: set[str] = set()
            for t_us in times_us:
                bmp = mmr.getFrameAtTime(t_us, MediaMetadataRetriever.OPTION_CLOSEST)
                if bmp is None:
                    bmp = mmr.getFrameAtTime(t_us, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                if bmp is None:
                    continue
                bos = ByteArrayOutputStream()
                bmp.compress(BitmapCompressFormat.PNG, 90, bos)
                raw_png = bytes(bos.toByteArray())
                if not raw_png:
                    continue
                with Image.open(io.BytesIO(raw_png)) as im:
                    im_rgb = im.convert("RGB")
                    w, h = im_rgb.size
                    if max(w, h) > max_edge:
                        scale = max_edge / float(max(w, h))
                        im_rgb = im_rgb.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.Resampling.BILINEAR)
                    out_io = io.BytesIO()
                    im_rgb.save(out_io, format="PNG", optimize=False)
                    png_bytes = out_io.getvalue()
                fh = hashlib.sha256(png_bytes).hexdigest()[:32]
                if fh in seen_hashes and len(frames_out) >= min_frames:
                    continue
                seen_hashes.add(fh)
                fname = f"{fh}.png"
                (media_p / fname).write_bytes(png_bytes)
                frames_out.append(fname)
        finally:
            try:
                mmr.release()
            except Exception:
                pass
    except Exception:
        pass

    aud_name: str | None = None
    aud_cand = media_p / f"{hashlib.sha256((stem + ':audio').encode()).hexdigest()[:32]}.wav"
    if decode_audio_to_wav16k(video_p, aud_cand, max_seconds=120.0):
        aud_name = aud_cand.name

    if frames_out:
        try:
            meta_lines = [f"F:{fn}" for fn in frames_out] + ([f"A:{aud_name}"] if aud_name else [])
            cached_meta.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
        except Exception:
            pass
    return frames_out, aud_name
