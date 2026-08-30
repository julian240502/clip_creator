import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def sample_video(tmp_path: Path) -> Path:
    """Petit clip 640x360 / 3 s avec une piste audio, généré par FFmpeg."""
    output = tmp_path / "sample.mp4"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(output),
    ]
    subprocess.run(command, check=True)
    return output
