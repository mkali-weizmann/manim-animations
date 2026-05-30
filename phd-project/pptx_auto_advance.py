#!/usr/bin/env python3
"""
pptx_auto_advance.py

For each slide number given on the command line, finds the embedded video on
that slide, reads its duration with ffprobe, and enables the PowerPoint
"Advance Slide -> After:" transition so the slide advances automatically.

Usage:
    python pptx_auto_advance.py presentation.pptx 3 7 12
    python pptx_auto_advance.py presentation.pptx 3 7 12 -o output.pptx

The input file is never modified in place; a new file is always written.
Slide numbering is 1-indexed, matching PowerPoint's own numbering.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from lxml import etree
except ImportError:
    sys.exit("lxml is required.  Install it with:  pip install lxml")

# ── namespaces ────────────────────────────────────────────────────────────────
PPTX_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
R_NS    = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".m4v", ".webm"}


# ── helpers ───────────────────────────────────────────────────────────────────

def get_slide_file_map(pptx_dir: Path) -> dict[int, Path]:
    """Return {1: Path, 2: Path, …} mapping logical slide order to XML paths."""
    rels_path = pptx_dir / "ppt" / "_rels" / "presentation.xml.rels"
    rid_to_path: dict[str, Path] = {}
    for rel in etree.parse(str(rels_path)).getroot():
        rid    = rel.get("Id", "")
        target = rel.get("Target", "")
        if "slides/slide" in target and "slideLayout" not in target:
            rid_to_path[rid] = (pptx_dir / "ppt" / target).resolve()

    prs_root  = etree.parse(str(pptx_dir / "ppt" / "presentation.xml")).getroot()
    sld_id_lst = prs_root.find(f"{{{PPTX_NS}}}sldIdLst")

    slide_map: dict[int, Path] = {}
    if sld_id_lst is not None:
        for i, sld_id in enumerate(sld_id_lst, start=1):
            rid = sld_id.get(f"{{{R_NS}}}id", "")
            if rid in rid_to_path:
                slide_map[i] = rid_to_path[rid]
    return slide_map


def find_slide_video(slide_xml: Path) -> Path | None:
    """Return the first video file referenced by this slide, or None."""
    rels_path = slide_xml.parent / "_rels" / (slide_xml.name + ".rels")
    if not rels_path.exists():
        return None
    for rel in etree.parse(str(rels_path)).getroot():
        target = rel.get("Target", "")
        if "../media/" in target:
            # Target is relative to the slide's directory
            media = (slide_xml.parent / target).resolve()
            if media.exists() and media.suffix.lower() in VIDEO_EXTENSIONS:
                return media
    return None


def get_video_duration_ms(video: Path) -> int | None:
    """Return video duration in milliseconds via ffprobe, or None on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout
        for stream in json.loads(out).get("streams", []):
            if stream.get("codec_type") == "video":
                return int(float(stream["duration"]) * 1000)
    except Exception as exc:
        print(f"  Warning: ffprobe failed for {video.name}: {exc}")
    return None


def enable_auto_advance(slide_xml: Path, duration_ms: int) -> None:
    """Set advTm=duration_ms on the slide's <p:transition>, creating it if absent."""
    tree = etree.parse(str(slide_xml))
    root = tree.getroot()
    tag  = f"{{{PPTX_NS}}}transition"

    transition = root.find(tag)          # direct child only (not recursive)
    if transition is None:
        transition = etree.Element(tag)
        timing = root.find(f"{{{PPTX_NS}}}timing")
        if timing is not None:
            timing.addprevious(transition)   # insert before <p:timing>
        else:
            root.append(transition)

    transition.set("advTm", str(duration_ms))
    # Note: advClick is deliberately left untouched so click-to-advance
    # behaviour is preserved if it was already set.

    tree.write(
        str(slide_xml),
        xml_declaration=True,
        encoding="UTF-8",
        standalone=True,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def process(pptx_path: Path, slide_numbers: list[int], output_path: Path) -> None:
    with TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)

        print(f"Extracting {pptx_path.name} …")
        with zipfile.ZipFile(pptx_path, "r") as z:
            z.extractall(tmp)

        slide_map = get_slide_file_map(tmp)

        for num in slide_numbers:
            print(f"\nSlide {num}:")
            if num not in slide_map:
                print(f"  Slide {num} not found in presentation (max = {max(slide_map)}) — skipping.")
                continue

            slide_xml = slide_map[num]
            video     = find_slide_video(slide_xml)
            if video is None:
                print("  No video found — skipping.")
                continue
            print(f"  Video    : {video.name}")

            ms = get_video_duration_ms(video)
            if ms is None:
                print("  Duration unknown — skipping.")
                continue
            print(f"  Duration : {ms / 1000:.3f} s  ({ms} ms)")

            enable_auto_advance(slide_xml, ms)
            print(f"  advTm set to {ms} ms  ✓")

        print(f"\nRepackaging → {output_path.name} …")
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in tmp.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(tmp))

    print("Done.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("pptx",   type=Path, help=".pptx file to process")
    ap.add_argument("slides", type=int,  nargs="+",
                    help="Slide numbers to update (1-indexed, as shown in PowerPoint)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output path (default: <name>_fixed.pptx)")
    args = ap.parse_args()

    if not args.pptx.exists():
        sys.exit(f"File not found: {args.pptx}")

    output = args.output or args.pptx.parent / (args.pptx.stem + "_fixed.pptx")
    process(args.pptx, args.slides, output)


if __name__ == "__main__":
    main()
