import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mido

from .config import SETTINGS
from .utils import clamp, run_cmd, write_json


@dataclass
class NoteEvent:
    start: float
    duration: float
    pitch: int
    velocity: int


@dataclass
class TabNote:
    timeStart: float
    duration: float
    pitch: int
    string: int
    fret: int


def parse_tuning(tuning: str, capo: int) -> Tuple[List[int], Optional[str]]:
    standard = [40, 45, 50, 55, 59, 64]  # E2 A2 D3 G3 B3 E4
    tuning = (tuning or "").strip().upper()
    if tuning == "EADGBE":
        return [p + capo for p in standard], None
    note_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    notes: List[str] = []
    idx = 0
    while idx < len(tuning):
        char = tuning[idx]
        if char not in note_map:
            return [p + capo for p in standard], "Tuning invalide, retour à EADGBE."
        if idx + 1 < len(tuning) and tuning[idx + 1] in ("#", "B"):
            notes.append(char + tuning[idx + 1])
            idx += 2
        else:
            notes.append(char)
            idx += 1
    if len(notes) != 6:
        return [p + capo for p in standard], "Tuning invalide, retour à EADGBE."
    octaves = [2, 2, 3, 3, 3, 4]
    midi: List[int] = []
    for note, octave in zip(notes, octaves):
        semitone = note_map[note[0]]
        if len(note) == 2 and note[1] == "#":
            semitone += 1
        elif len(note) == 2 and note[1] == "B":
            semitone -= 1
        midi.append((octave + 1) * 12 + semitone + capo)
    return midi, None


def ensure_dependencies(require_youtube: bool = False) -> None:
    missing = []
    for tool in (
        SETTINGS.ffmpeg_path,
        SETTINGS.ffprobe_path,
        SETTINGS.basic_pitch_path,
        SETTINGS.demucs_path,
        SETTINGS.musescore_path,
    ):
        from shutil import which

        if which(tool) is None:
            missing.append(tool)
    if require_youtube:
        from shutil import which

        if which(SETTINGS.yt_dlp_path) is None:
            missing.append(SETTINGS.yt_dlp_path)
    if missing:
        raise RuntimeError(
            "Dépendances manquantes: "
            + ", ".join(missing)
            + ". Installez-les (ffmpeg, demucs, basic-pitch, yt-dlp si YouTube)."
        )


def ffprobe_duration(path: str) -> float:
    import subprocess

    result = subprocess.run(
        [
            SETTINGS.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Impossible de lire la durée audio.")
    return float(result.stdout.strip())


def convert_to_wav(input_path: str, output_path: str, logger) -> None:
    run_cmd(
        [
            SETTINGS.ffmpeg_path,
            "-y",
            "-nostdin",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            str(SETTINGS.sample_rate),
            "-sample_fmt",
            "s16",
            output_path,
        ],
        logger,
    )


def trim_wav(
    input_path: str,
    output_path: str,
    start_seconds: float,
    end_seconds: Optional[float],
    logger,
) -> None:
    args = [
        SETTINGS.ffmpeg_path,
        "-y",
        "-nostdin",
        "-ss",
        str(start_seconds),
    ]
    if end_seconds is not None:
        args += ["-to", str(end_seconds)]
    args += [
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        str(SETTINGS.sample_rate),
        "-sample_fmt",
        "s16",
        output_path,
    ]
    run_cmd(args, logger)


def download_youtube_audio(url: str, output_dir: str, logger) -> str:
    output_template = os.path.join(output_dir, "youtube.%(ext)s")
    run_cmd(
        [
            SETTINGS.yt_dlp_path,
            "-f",
            "bestaudio/best",
            "--no-playlist",
            "--extract-audio",
            "--audio-format",
            "wav",
            "--output",
            output_template,
            url,
        ],
        logger,
    )
    wav_path = os.path.join(output_dir, "youtube.wav")
    if not os.path.exists(wav_path):
        raise RuntimeError("Téléchargement YouTube échoué.")
    return wav_path


def run_demucs(input_wav: str, output_dir: str, logger) -> str:
    run_cmd(
        [
            SETTINGS.demucs_path,
            "-n",
            "htdemucs",
            "-o",
            output_dir,
            input_wav,
        ],
        logger,
        timeout_seconds=SETTINGS.subprocess_timeout_seconds,
    )
    basename = os.path.splitext(os.path.basename(input_wav))[0]
    other = os.path.join(output_dir, "htdemucs", basename, "other.wav")
    if not os.path.exists(other):
        raise RuntimeError("Demucs n'a pas produit other.wav.")
    return other


def run_basic_pitch(input_wav: str, output_dir: str, logger) -> str:
    run_cmd(
        [
            SETTINGS.basic_pitch_path,
            output_dir,
            input_wav,
        ],
        logger,
        timeout_seconds=SETTINGS.subprocess_timeout_seconds,
    )
    for name in os.listdir(output_dir):
        if name.endswith(".mid") or name.endswith(".midi"):
            return os.path.join(output_dir, name)
    raise RuntimeError("Basic Pitch n'a pas généré de MIDI.")


def load_midi_notes(midi_path: str) -> Tuple[List[NoteEvent], Optional[float]]:
    midi = mido.MidiFile(midi_path)
    tempo = 500000
    ticks_per_beat = midi.ticks_per_beat
    events: List[NoteEvent] = []
    note_on: Dict[Tuple[int, int], Tuple[float, int]] = {}
    current_time = 0.0
    for msg in mido.merge_tracks(midi.tracks):
        current_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        if msg.type == "note_on" and msg.velocity > 0:
            note_on[(msg.channel, msg.note)] = (current_time, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            key = (msg.channel, msg.note)
            if key in note_on:
                start, velocity = note_on.pop(key)
                duration = max(0.01, current_time - start)
                events.append(NoteEvent(start=start, duration=duration, pitch=msg.note, velocity=velocity))
    bpm = None
    if tempo:
        bpm = 60_000_000 / tempo
    return events, bpm


def post_process_notes(
    notes: List[NoteEvent],
    quality: str,
) -> Tuple[List[NoteEvent], Dict[str, float]]:
    if not notes:
        return [], {"short_ratio": 1.0, "out_of_range_ratio": 1.0}
    quality = quality.lower()
    subdiv = 2 if quality == "fast" else 4
    seconds_per_beat = 60.0 / SETTINGS.default_tempo_bpm
    grid = seconds_per_beat / subdiv
    min_duration = 0.1 if quality == "fast" else 0.05
    processed: List[NoteEvent] = []
    short_notes = 0
    out_of_range = 0
    for note in notes:
        start = round(note.start / grid) * grid
        duration = round(note.duration / grid) * grid
        duration = max(duration, grid)
        if duration < min_duration:
            short_notes += 1
            continue
        if note.pitch < 40 or note.pitch > 88:
            out_of_range += 1
            continue
        processed.append(NoteEvent(start=start, duration=duration, pitch=note.pitch, velocity=note.velocity))
    total = max(1, len(notes))
    return processed, {
        "short_ratio": short_notes / total,
        "out_of_range_ratio": out_of_range / total,
    }


def group_chords(notes: List[NoteEvent], threshold: float = 0.03) -> List[List[NoteEvent]]:
    if not notes:
        return []
    notes_sorted = sorted(notes, key=lambda n: n.start)
    groups: List[List[NoteEvent]] = []
    current: List[NoteEvent] = [notes_sorted[0]]
    for note in notes_sorted[1:]:
        if abs(note.start - current[-1].start) <= threshold:
            current.append(note)
        else:
            groups.append(current)
            current = [note]
    groups.append(current)
    return groups


def possible_positions(pitch: int, open_strings: List[int], max_fret: int = 24) -> List[Tuple[int, int]]:
    positions = []
    for idx, open_pitch in enumerate(open_strings):
        fret = pitch - open_pitch
        if 0 <= fret <= max_fret:
            positions.append((idx, fret))
    return positions


def assign_chord(
    notes: List[NoteEvent],
    open_strings: List[int],
    last_positions: Dict[int, int],
) -> Dict[int, Tuple[int, int]]:
    best_cost = float("inf")
    best_assignment: Dict[int, Tuple[int, int]] = {}

    candidates = [possible_positions(n.pitch, open_strings) for n in notes]
    if any(len(c) == 0 for c in candidates):
        return {}

    def recurse(i: int, used_strings: set, current_cost: float, assignment: Dict[int, Tuple[int, int]]):
        nonlocal best_cost, best_assignment
        if i >= len(notes):
            if current_cost < best_cost:
                best_cost = current_cost
                best_assignment = dict(assignment)
            return
        if current_cost >= best_cost:
            return
        for string_idx, fret in candidates[i]:
            if string_idx in used_strings:
                continue
            last_fret = last_positions.get(string_idx)
            move_cost = abs(fret - last_fret) if last_fret is not None else 0
            penalty = 5 if fret > 20 else 0
            total = current_cost + move_cost + penalty
            assignment[i] = (string_idx, fret)
            recurse(i + 1, used_strings | {string_idx}, total, assignment)
            assignment.pop(i, None)

    recurse(0, set(), 0.0, {})
    return best_assignment


def map_notes_to_tab(
    notes: List[NoteEvent],
    tuning: str,
    capo: int,
) -> Tuple[List[TabNote], Optional[str]]:
    open_strings, tuning_warning = parse_tuning(tuning, capo)
    chord_groups = group_chords(notes)
    tab_notes: List[TabNote] = []
    last_positions: Dict[int, int] = {}
    for group in chord_groups:
        assignment = assign_chord(group, open_strings, last_positions)
        if not assignment:
            for note in group:
                candidates = possible_positions(note.pitch, open_strings)
                if not candidates:
                    continue
                string_idx, fret = min(
                    candidates,
                    key=lambda pair: (abs(pair[1] - last_positions.get(pair[0], pair[1])), pair[1]),
                )
                last_positions[string_idx] = fret
                tab_notes.append(
                    TabNote(
                        timeStart=note.start,
                        duration=note.duration,
                        pitch=note.pitch,
                        string=6 - string_idx,
                        fret=fret,
                    )
                )
            continue
        for note_idx, (string_idx, fret) in assignment.items():
            note = group[note_idx]
            last_positions[string_idx] = fret
            tab_notes.append(
                TabNote(
                    timeStart=note.start,
                    duration=note.duration,
                    pitch=note.pitch,
                    string=6 - string_idx,
                    fret=fret,
                )
            )
    return sorted(tab_notes, key=lambda n: n.timeStart), tuning_warning


def compute_confidence(metrics: Dict[str, float], notes: List[NoteEvent]) -> float:
    if not notes:
        return 0.0
    density = len(notes) / max(1.0, max(n.start + n.duration for n in notes))
    density_penalty = clamp(density / 8.0, 0.0, 1.0)
    score = 1.0 - (metrics["short_ratio"] * 0.4 + metrics["out_of_range_ratio"] * 0.4 + density_penalty * 0.2)
    return clamp(score, 0.0, 1.0)


def render_tab_json(
    tab_notes: List[TabNote],
    tuning: str,
    capo: int,
    quality: str,
    tempo_bpm: float,
    tempo_source: str,
    warnings: List[str],
) -> Dict:
    total_duration = max((n.timeStart + n.duration for n in tab_notes), default=0.0)
    seconds_per_beat = 60.0 / tempo_bpm
    measure_duration = seconds_per_beat * 4
    measures = []
    measure_count = int(math.ceil(total_duration / measure_duration)) if total_duration else 1
    for idx in range(measure_count):
        start = idx * measure_duration
        measures.append(
            {
                "index": idx + 1,
                "startTime": start,
                "endTime": start + measure_duration,
                "beats": [
                    {
                        "index": b + 1,
                        "startTime": start + b * seconds_per_beat,
                        "endTime": start + (b + 1) * seconds_per_beat,
                    }
                    for b in range(4)
                ],
            }
        )
    return {
        "tempo": {"bpm": tempo_bpm, "source": tempo_source},
        "tuning": tuning,
        "capo": capo,
        "quality": quality,
        "warnings": warnings,
        "notes": [note.__dict__ for note in tab_notes],
        "measures": measures,
    }


def render_tab_txt(tab_notes: List[TabNote], tempo_bpm: float, quality: str) -> str:
    if not tab_notes:
        return ""
    seconds_per_beat = 60.0 / tempo_bpm
    subdiv = 2 if quality == "fast" else 4
    step = seconds_per_beat / subdiv
    total_duration = max(n.timeStart + n.duration for n in tab_notes)
    steps = int(math.ceil(total_duration / step)) + 1

    def blank_row():
        return ["---" for _ in range(steps)]

    rows = {s: blank_row() for s in range(1, 7)}
    for note in tab_notes:
        idx = int(round(note.timeStart / step))
        if idx < 0 or idx >= steps:
            continue
        fret_str = str(note.fret)
        token = (fret_str + "-" * (3 - len(fret_str)))[:3]
        rows[note.string][idx] = token

    line_order = [1, 2, 3, 4, 5, 6]
    names = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}
    lines = []
    for string in line_order:
        lines.append(f"{names[string]}|" + "".join(rows[string]) + "|")
    return "\n".join(lines)


def write_tab_outputs(
    tab_notes: List[TabNote],
    tuning: str,
    capo: int,
    quality: str,
    tempo_bpm: float,
    tempo_source: str,
    warnings: List[str],
    tab_txt_path: str,
    tab_json_path: str,
) -> None:
    tab_txt = render_tab_txt(tab_notes, tempo_bpm, quality)
    if not tab_txt:
        raise RuntimeError("Aucune tablature générée (pas de notes exploitables).")
    with open(tab_txt_path, "w", encoding="utf-8") as f:
        f.write(tab_txt)
    tab_json = render_tab_json(tab_notes, tuning, capo, quality, tempo_bpm, tempo_source, warnings)
    write_json(tab_json_path, tab_json)


def _midi_to_pitch(midi: int) -> Tuple[str, int, int]:
    steps = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
    alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    step = steps[midi % 12]
    alter = alters[midi % 12]
    octave = midi // 12 - 1
    return step, alter, octave


def _duration_pieces(duration_div: int, divisions: int) -> List[int]:
    if duration_div <= 0:
        return []
    whole = divisions * 4
    half = divisions * 2
    quarter = divisions
    eighth = max(1, divisions // 2)
    sixteenth = max(1, divisions // 4)
    values = [whole, half, quarter, eighth, sixteenth]
    pieces: List[int] = []
    remaining = duration_div
    while remaining > 0:
        for value in values:
            if value <= remaining:
                pieces.append(value)
                remaining -= value
                break
    return pieces


def _duration_type(duration_div: int, divisions: int) -> str:
    mapping = {
        divisions * 4: "whole",
        divisions * 2: "half",
        divisions: "quarter",
        max(1, divisions // 2): "eighth",
        max(1, divisions // 4): "16th",
    }
    return mapping.get(duration_div, "quarter")


def _format_tuning_label(open_strings: List[int]) -> str:
    names: List[str] = []
    for idx, midi in enumerate(reversed(open_strings), start=1):
        step, alter, _ = _midi_to_pitch(midi)
        note = step + ("#" if alter == 1 else "")
        names.append(f"{idx} - {note}")
    return "Accordage: " + ", ".join(names)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def write_tab_musicxml(
    tab_notes: List[TabNote],
    tuning: str,
    capo: int,
    tempo_bpm: float,
    title: str,
    artist: str,
    annotations: Optional[List[Dict[str, str]]],
    output_path: str,
) -> None:
    open_strings, _ = parse_tuning(tuning, capo)
    divisions = 4
    seconds_per_beat = 60.0 / tempo_bpm
    total_duration = max((n.timeStart + n.duration for n in tab_notes), default=0.0)
    total_beats = total_duration / seconds_per_beat if seconds_per_beat else 0.0
    measure_div = divisions * 4
    measure_count = max(1, int(math.ceil(total_beats / 4.0))) if total_beats else 1

    events: Dict[Tuple[int, int], Dict[str, object]] = {}
    for note in tab_notes:
        start_div = int(round(note.timeStart / seconds_per_beat * divisions))
        duration_div = max(1, int(round(note.duration / seconds_per_beat * divisions)))
        remaining = duration_div
        current_start = start_div
        segment_index = 0
        while remaining > 0:
            measure_idx = current_start // measure_div
            offset = current_start % measure_div
            available = measure_div - offset
            segment_duration = min(remaining, available)
            has_more = remaining > segment_duration
            if duration_div > segment_duration:
                if segment_index == 0 and has_more:
                    tie_start = True
                    tie_stop = False
                elif has_more:
                    tie_start = True
                    tie_stop = True
                else:
                    tie_start = False
                    tie_stop = True
            else:
                tie_start = False
                tie_stop = False
            key = (measure_idx, offset)
            if key not in events:
                events[key] = {"notes": [], "duration": segment_duration, "tie_start": False, "tie_stop": False}
            events[key]["notes"].append(note)
            events[key]["duration"] = max(int(events[key]["duration"]), segment_duration)
            events[key]["tie_start"] = bool(events[key]["tie_start"]) or tie_start
            events[key]["tie_stop"] = bool(events[key]["tie_stop"]) or tie_stop
            remaining -= segment_duration
            current_start += segment_duration
            segment_index += 1

    annotations = annotations or []
    annotations_by_measure: Dict[int, List[Dict[str, str]]] = {}
    for item in annotations:
        try:
            ts = float(item.get("timeStart", 0))
        except (TypeError, ValueError):
            ts = 0.0
        idx = int((ts / seconds_per_beat) // 4) if seconds_per_beat else 0
        annotations_by_measure.setdefault(idx, []).append(item)

    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title or "Transcription"
    identification = ET.SubElement(score, "identification")
    ET.SubElement(identification, "creator", type="composer").text = artist or "Artiste inconnu"
    defaults = ET.SubElement(score, "defaults")
    page_layout = ET.SubElement(defaults, "page-layout")
    ET.SubElement(page_layout, "page-height").text = "1683.78"
    ET.SubElement(page_layout, "page-width").text = "1190.55"
    margins = ET.SubElement(page_layout, "page-margins", type="both")
    ET.SubElement(margins, "left-margin").text = "70"
    ET.SubElement(margins, "right-margin").text = "70"
    ET.SubElement(margins, "top-margin").text = "70"
    ET.SubElement(margins, "bottom-margin").text = "70"

    credit = ET.SubElement(score, "credit")
    credit_words = ET.SubElement(
        credit,
        "credit-words",
        default_x="1120",
        default_y="1600",
        justify="right",
    )
    credit_words.text = _format_tuning_label(open_strings)

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Guitare"

    part = ET.SubElement(score, "part", id="P1")
    section_labels = ["A", "B", "C", "D", "E"]

    for measure_idx in range(measure_count):
        measure = ET.SubElement(part, "measure", number=str(measure_idx + 1))
        if measure_idx == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(divisions)
            key = ET.SubElement(attributes, "key")
            ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = "4"
            ET.SubElement(time, "beat-type").text = "4"
            clef = ET.SubElement(attributes, "clef")
            ET.SubElement(clef, "sign").text = "TAB"
            ET.SubElement(clef, "line").text = "5"
            staff_details = ET.SubElement(attributes, "staff-details")
            ET.SubElement(staff_details, "staff-lines").text = "6"
            for line_idx, midi in enumerate(reversed(open_strings), start=1):
                step, alter, octave = _midi_to_pitch(midi)
                tuning_el = ET.SubElement(staff_details, "staff-tuning", line=str(line_idx))
                ET.SubElement(tuning_el, "tuning-step").text = step
                if alter:
                    ET.SubElement(tuning_el, "tuning-alter").text = str(alter)
                ET.SubElement(tuning_el, "tuning-octave").text = str(octave)
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = str(int(round(tempo_bpm)))
            ET.SubElement(direction, "sound", tempo=str(int(round(tempo_bpm))))

        if measure_idx < len(section_labels):
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            ET.SubElement(direction_type, "rehearsal").text = section_labels[measure_idx]

        for annotation in annotations_by_measure.get(measure_idx, []):
            text = (annotation.get("text") or "").strip()
            if not text:
                continue
            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            ET.SubElement(direction_type, "words").text = text

        measure_events = [
            (offset, events[(measure_idx, offset)])
            for (idx, offset) in events.keys()
            if idx == measure_idx
        ]
        measure_events.sort(key=lambda item: item[0])
        items: List[Dict[str, object]] = []
        cursor = 0
        for offset, payload in measure_events:
            duration = int(payload["duration"])
            if offset > cursor:
                items.append({"kind": "rest", "duration": offset - cursor})
            remaining = max(0, measure_div - offset)
            duration = min(duration, remaining)
            if duration > 0:
                items.append(
                    {
                        "kind": "event",
                        "duration": duration,
                        "notes": payload["notes"],
                        "tie_start": payload.get("tie_start", False),
                        "tie_stop": payload.get("tie_stop", False),
                    }
                )
                cursor = offset + duration
        if cursor < measure_div:
            items.append({"kind": "rest", "duration": measure_div - cursor})

        beam_status: Dict[int, str] = {}
        group: List[int] = []
        for idx, item in enumerate(items):
            if item["kind"] == "event" and int(item["duration"]) <= max(1, divisions // 2):
                group.append(idx)
            else:
                if len(group) >= 2:
                    beam_status[group[0]] = "begin"
                    for mid in group[1:-1]:
                        beam_status[mid] = "continue"
                    beam_status[group[-1]] = "end"
                group = []
        if len(group) >= 2:
            beam_status[group[0]] = "begin"
            for mid in group[1:-1]:
                beam_status[mid] = "continue"
            beam_status[group[-1]] = "end"

        for item_idx, item in enumerate(items):
            duration = int(item["duration"])
            pieces = _duration_pieces(duration, divisions)
            if item["kind"] == "rest":
                for piece in pieces:
                    note_el = ET.SubElement(measure, "note")
                    ET.SubElement(note_el, "rest")
                    ET.SubElement(note_el, "duration").text = str(piece)
                    ET.SubElement(note_el, "type").text = _duration_type(piece, divisions)
                continue

            notes: List[TabNote] = item["notes"]  # type: ignore[assignment]
            base_tie_start = bool(item.get("tie_start"))
            base_tie_stop = bool(item.get("tie_stop"))
            for p_idx, piece in enumerate(pieces):
                if len(pieces) > 1:
                    if p_idx == 0:
                        tie_start = True
                        tie_stop = False
                    elif p_idx == len(pieces) - 1:
                        tie_start = False
                        tie_stop = True
                    else:
                        tie_start = True
                        tie_stop = True
                else:
                    tie_start = False
                    tie_stop = False
                if base_tie_start and p_idx == 0:
                    tie_start = True
                if base_tie_stop and p_idx == len(pieces) - 1:
                    tie_stop = True
                for n_idx, tab_note in enumerate(notes):
                    note_el = ET.SubElement(measure, "note")
                    if n_idx > 0:
                        ET.SubElement(note_el, "chord")
                    step, alter, octave = _midi_to_pitch(tab_note.pitch)
                    pitch_el = ET.SubElement(note_el, "pitch")
                    ET.SubElement(pitch_el, "step").text = step
                    if alter:
                        ET.SubElement(pitch_el, "alter").text = str(alter)
                    ET.SubElement(pitch_el, "octave").text = str(octave)
                    ET.SubElement(note_el, "duration").text = str(piece)
                    ET.SubElement(note_el, "type").text = _duration_type(piece, divisions)
                    ET.SubElement(note_el, "voice").text = "1"
                    if item_idx in beam_status and n_idx == 0 and p_idx == 0:
                        ET.SubElement(note_el, "beam", number="1").text = beam_status[item_idx]
                    notations = ET.SubElement(note_el, "notations")
                    if tie_start:
                        ET.SubElement(notations, "tied", type="start")
                        ET.SubElement(note_el, "tie", type="start")
                    if tie_stop:
                        ET.SubElement(notations, "tied", type="stop")
                        ET.SubElement(note_el, "tie", type="stop")
                    technical = ET.SubElement(notations, "technical")
                    ET.SubElement(technical, "string").text = str(tab_note.string)
                    ET.SubElement(technical, "fret").text = str(tab_note.fret)

    _indent_xml(score)
    tree = ET.ElementTree(score)
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
            b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
        )
        tree.write(f, encoding="utf-8")


def render_musicxml_to_pdf(musicxml_path: str, pdf_path: str, logger) -> None:
    run_cmd(
        [
            SETTINGS.musescore_path,
            "-o",
            pdf_path,
            musicxml_path,
        ],
        logger,
        env={
            "QT_QPA_PLATFORM": "offscreen",
            "DISPLAY": ":0",
        },
    )

