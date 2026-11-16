"""Small Pillow GIF/contact-sheet exports, without video codecs or a display."""

from dataclasses import dataclass
from html import escape
import math
from pathlib import Path

import numpy as np


@dataclass
class Panel:
    title: str
    frames: list
    steps: list
    fixed: bool = False
    details: list | None = None


def sampling(length, every):
    if type(length) is not int or length < 1 or type(every) is not int or every < 1:
        raise ValueError("Frame count and sample stride must be positive integers")
    selected = list(range(0, length, every))
    if selected[-1] != length-1:
        selected.append(length-1)
    return selected


def font(size):
    from PIL import ImageFont
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def compose(panels, index, scale):
    from PIL import Image, ImageDraw
    images = []
    for panel in panels:
        if not panel.frames or len(panel.frames) != len(panel.steps):
            raise ValueError("Panel frames and step labels must be nonempty and aligned")
        position = min(index, len(panel.frames)-1)
        frame = np.asarray(panel.frames[position])
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError("Preview expects RGB uint8 frames")
        image = Image.fromarray(frame).resize((frame.shape[1]*scale, frame.shape[0]*scale), Image.Resampling.NEAREST)
        tile = Image.new("RGB", (image.width, image.height+66), "white")
        tile.paste(image, (0, 66))
        draw = ImageDraw.Draw(tile)
        draw.text((6, 3), panel.title, fill="black", font=font(14 if scale == 1 else 17))
        held = " (fixed)" if panel.fixed else " (held at end)" if index >= len(panel.frames) else ""
        draw.text((6, 25), f"raw step {panel.steps[position]}{held}", fill="#333333", font=font(12 if scale == 1 else 15))
        if panel.details:
            draw.text((6, 45), panel.details[position], fill="#333333", font=font(11 if scale == 1 else 14))
        images.append(tile)
    if len({image.height for image in images}) != 1:
        raise ValueError("All preview panels must have the same height")
    output = Image.new("RGB", (sum(image.width for image in images), images[0].height), "#dddddd")
    cursor = 0
    for image in images:
        output.paste(image, (cursor, 0))
        cursor += image.width
    return output


def save_gallery(directory, stem, panels, *, fps, every=1, scale=2):
    """The first panel sets the timeline. Short references are explicitly held."""
    from PIL import Image
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if type(scale) is not int or not 1 <= scale <= 4:
        raise ValueError("scale must be an integer from 1 to 4")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    selected = sampling(len(panels[0].frames), every)
    animation = [compose(panels, index, scale) for index in selected]
    # Use the raw-frame spacing to preserve playback speed when subsampling.
    durations = [max(10, int(round((right-left)*1000/fps))) for left, right in zip(selected, selected[1:])]
    durations.append(max(10, int(round(1000/fps))))
    gif = directory/f"{stem}.gif"
    animation[0].save(gif, save_all=True, append_images=animation[1:], duration=durations,
                      loop=0, disposal=2, optimize=True)
    positions = np.unique(np.linspace(0, len(panels[0].frames)-1, min(6, len(panels[0].frames)), dtype=int)).tolist()
    tiles = [compose(panels, index, 1) for index in positions]
    columns = min(3, len(tiles))
    rows = (len(tiles)+columns-1)//columns
    sheet = Image.new("RGB", (tiles[0].width*columns, tiles[0].height*rows), "#dddddd")
    for position, tile in enumerate(tiles):
        sheet.paste(tile, ((position % columns)*tile.width, (position//columns)*tile.height))
    png = directory/f"{stem}-contact-sheet.png"
    sheet.save(png)
    return {"gif": gif.name, "contact_sheet": png.name, "raw_frames": len(panels[0].frames),
            "displayed_frames": len(selected), "sample_every": every, "raw_fps": fps,
            "playback_milliseconds": sum(durations), "render": "simulator/stored RGB; no latent decoder"}


def index_page(directory, title, galleries, description):
    directory = Path(directory)
    sections = []
    for label, item in galleries:
        gif, png = escape(item["gif"], quote=True), escape(item["contact_sheet"], quote=True)
        sections.append(f'<h2>{escape(label)}</h2><p><a href="{gif}">Animation</a> · <a href="{png}">Contact sheet</a></p>'
                        f'<img src="{gif}" alt="{escape(label, quote=True)}">')
    page = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<style>body{font-family:system-ui;margin:24px;max-width:1500px}img{max-width:100%;border:1px solid #ddd}</style>'
            f'<title>{escape(title)}</title><h1>{escape(title)}</h1><p>{escape(description)}</p>'
            + "".join(sections))
    (directory/"index.html").write_text(page)
