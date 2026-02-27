import os
import subprocess


def concatenate_files(chunk_files: list[str], output_path: str) -> None:
    """Route to the appropriate concatenation strategy based on file type."""
    if not chunk_files:
        return

    ext = os.path.splitext(output_path)[1].lower()
    match ext:
        case ".mp4" | ".avi" | ".mov":
            _concatenate_videos_ffmpeg(chunk_files, output_path)
        case ".csv":
            _concatenate_csvs(chunk_files, output_path)
        case _:
            raise ValueError(f"No concatenation strategy for file type: {ext}")


def _concatenate_videos_ffmpeg(chunk_files: list[str], output_path: str) -> None:
    """Lossless concatenation via ffmpeg stream copy — no re-encoding."""
    list_path = output_path + "_chunks.txt"
    with open(list_path, "w") as f:
        for path in chunk_files:
            f.write(f"file '{os.path.abspath(path)}'\n")

    result = subprocess.run(
        [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-c",
            "copy",
            output_path,
            "-y",
            "-loglevel",
            "error",
        ],
        capture_output=True,
        text=True,
    )
    os.remove(list_path)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concatenation failed: {result.stderr}")


def _concatenate_csvs(chunk_files: list[str], output_path: str) -> None:
    """Concatenate CSV chunks in order, writing header only once."""
    with open(output_path, "w") as out:
        for i, path in enumerate(chunk_files):
            with open(path) as chunk:
                lines = chunk.readlines()
                if not lines:
                    continue
                if i == 0:
                    out.writelines(lines)  # include header from first chunk
                else:
                    out.writelines(lines[1:])  # skip header on subsequent chunks
