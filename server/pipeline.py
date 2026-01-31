import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import mido

from .config import SETTINGS
from .utils import clamp, run_cmd, read_json, write_json


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


DEFAULT_DIVISIONS = 4
DEFAULT_TIME_SIGNATURE = "4/4"
GUITAR_WRITTEN_SHIFT = 12
LOW_PITCH_THRESHOLD = 55
HIGH_PITCH_THRESHOLD = 79


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
    tempo_bpm: float,
) -> Tuple[List[NoteEvent], Dict[str, float]]:
    if not notes:
        return [], {"short_ratio": 1.0, "out_of_range_ratio": 1.0}
    quality = quality.lower()
    subdiv = 2 if quality == "fast" else 4
    tempo_value = tempo_bpm if tempo_bpm else SETTINGS.default_tempo_bpm
    seconds_per_beat = 60.0 / tempo_value if tempo_value else 0.0
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


def _seconds_per_beat(tempo_bpm: float) -> float:
    tempo = float(tempo_bpm or SETTINGS.default_tempo_bpm)
    if tempo <= 0:
        tempo = SETTINGS.default_tempo_bpm
    return 60.0 / tempo


def _build_measure_events(
    entries: Iterable[Any],
    get_start: Callable[[Any], float],
    get_duration: Callable[[Any], float],
    serialize_note: Callable[[Any], Dict[str, Any]],
    seconds_per_beat: float,
    divisions: int,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    measure_ticks = divisions * 4
    events: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for entry in entries:
        start_ticks = get_start(entry)
        duration_secs = get_duration(entry)
        start_tick = int(round(start_ticks / seconds_per_beat * divisions))
        duration_tick = max(1, int(round(duration_secs / seconds_per_beat * divisions)))
        remaining = duration_tick
        current_start = start_tick
        segment_index = 0
        while remaining > 0:
            measure_idx = current_start // measure_ticks
            offset = current_start % measure_ticks
            available = measure_ticks - offset
            segment_duration = min(remaining, available)
            has_more = remaining > segment_duration
            if duration_tick > segment_duration:
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
            entry_payload = events.setdefault(
                key,
                {"notes": [], "duration": 0, "tie_start": False, "tie_stop": False},
            )
            entry_payload["notes"].append(serialize_note(entry))
            entry_payload["duration"] = max(int(entry_payload["duration"]), segment_duration)
            entry_payload["tie_start"] = bool(entry_payload["tie_start"]) or tie_start
            entry_payload["tie_stop"] = bool(entry_payload["tie_stop"]) or tie_stop
            remaining -= segment_duration
            current_start += segment_duration
            segment_index += 1
    return events


def _events_to_measures(events: Dict[Tuple[int, int], Dict[str, Any]], divisions: int) -> List[Dict[str, Any]]:
    measure_indices = sorted({idx for idx, _ in events.keys()})
    measure_count = max(measure_indices) + 1 if measure_indices else 1
    measures = []
    for measure_idx in range(measure_count):
        measure_events = [
            (offset, events[(measure_idx, offset)])
            for (idx, offset) in events.keys()
            if idx == measure_idx
        ]
        measure_events.sort(key=lambda item: item[0])
        event_list = []
        for offset, payload in measure_events:
            event_list.append(
                {
                    "startTick": offset,
                    "durationTick": payload["duration"],
                    "tieStart": payload.get("tie_start", False),
                    "tieStop": payload.get("tie_stop", False),
                    "notes": payload["notes"],
                }
            )
        measures.append({"number": measure_idx + 1, "events": event_list})
    return measures


def _build_measure_items(
    measure_events: List[Dict[str, Any]],
    divisions: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    measure_div = divisions * 4
    items: List[Dict[str, Any]] = []
    cursor = 0
    sorted_events = sorted(measure_events, key=lambda ev: int(ev.get("startTick", 0)))
    for event in sorted_events:
        offset = int(event.get("startTick", 0))
        if offset > cursor:
            items.append({"kind": "rest", "duration": offset - cursor})
        duration = int(event.get("durationTick", 0))
        duration = min(duration, max(0, measure_div - offset))
        if duration > 0:
            items.append(
                {
                    "kind": "event",
                    "duration": duration,
                    "notes": event.get("notes", []),
                    "tie_start": bool(event.get("tieStart")),
                    "tie_stop": bool(event.get("tieStop")),
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
    return items, beam_status


def _determine_written_octave_shift(notes: List[NoteEvent]) -> int:
    if not notes:
        return GUITAR_WRITTEN_SHIFT
    pitches = sorted(note.pitch for note in notes)
    median_idx = len(pitches) // 2
    median_value = pitches[median_idx]
    if median_value > HIGH_PITCH_THRESHOLD:
        return -GUITAR_WRITTEN_SHIFT
    if median_value < LOW_PITCH_THRESHOLD:
        return GUITAR_WRITTEN_SHIFT
    return GUITAR_WRITTEN_SHIFT


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


def pitch_from_string_fret(string_number: int, fret: int, open_strings: List[int]) -> int:
    if not (1 <= string_number <= len(open_strings)):
        raise ValueError(f"Numéro de corde invalide: {string_number}")
    open_index = len(open_strings) - string_number
    return open_strings[open_index] + fret


def _midi_to_note_label(midi: int) -> str:
    step, alter, octave = _midi_to_pitch(midi)
    label = step
    if alter == 1:
        label += "#"
    elif alter == -1:
        label += "b"
    return f"{label}{octave}"


def build_tab_json(
    tab_notes: List[TabNote],
    tuning: str,
    capo: int,
    quality: str,
    tempo_bpm: float,
    tempo_source: str,
    warnings: List[str],
) -> Tuple[Dict[str, object], int]:
    divisions = DEFAULT_DIVISIONS
    seconds_per_beat = _seconds_per_beat(tempo_bpm)
    events = _build_measure_events(
        tab_notes,
        lambda note: note.timeStart,
        lambda note: note.duration,
        lambda note: {"string": note.string, "fret": note.fret, "midi": note.pitch},
        seconds_per_beat,
        divisions,
    )
    measures = _events_to_measures(events, divisions)
    open_strings, _ = parse_tuning(tuning, capo)

    tuning_names = [_midi_to_note_label(midi) for midi in reversed(open_strings)]
    metadata = {
        "tuning": tuning_names,
        "tempo": tempo_bpm,
        "tempoSource": tempo_source,
        "timeSignature": DEFAULT_TIME_SIGNATURE,
        "divisions": divisions,
        "quality": quality,
    }
    total_notes = sum(len(event["notes"]) for measure in measures for event in measure["events"])
    tab_json = {
        "metadata": metadata,
        "quality": quality,
        "warnings": warnings,
        "measures": measures,
    }
    return tab_json, total_notes


def build_score_json(
    notes: List[NoteEvent],
    quality: str,
    tempo_bpm: float,
    tempo_source: str,
) -> Tuple[Dict[str, object], int]:
    divisions = DEFAULT_DIVISIONS
    seconds_per_beat = _seconds_per_beat(tempo_bpm)
    written_shift = _determine_written_octave_shift(notes)

    def serialize(note: NoteEvent) -> Dict[str, int]:
        notated = int(clamp(note.pitch + written_shift, 0, 127))
        return {"midiPitch": notated}

    sorted_notes = sorted(notes, key=lambda note: note.start)
    events = _build_measure_events(
        sorted_notes,
        lambda note: note.start,
        lambda note: note.duration,
        serialize,
        seconds_per_beat,
        divisions,
    )
    measures = _events_to_measures(events, divisions)
    total_notes = sum(len(event["notes"]) for measure in measures for event in measure["events"])

    metadata = {
        "tempo": tempo_bpm,
        "tempoSource": tempo_source,
        "timeSignature": DEFAULT_TIME_SIGNATURE,
        "divisions": divisions,
        "quality": quality,
        "scoreWrittenOctaveShift": written_shift,
    }
    score_json = {
        "metadata": metadata,
        "measures": measures,
    }
    return score_json, total_notes


def render_tab_txt_from_json(
    tab_json: Dict[str, object],
    quality: str,
    measures_per_system: int = 2,
    logger=None,
) -> Tuple[str, int, List[str]]:
    metadata = tab_json.get("metadata", {})
    divisions = int(metadata.get("divisions") or 4)
    measure_ticks = divisions * 4
    measures = tab_json.get("measures", [])
    measure_count = max(len(measures), 1)
    measure_grids = [
        {string_num: ["---"] * measure_ticks for string_num in range(1, 7)}
        for _ in range(measure_count)
    ]

    warnings: List[str] = []
    note_count = 0
    for idx, measure in enumerate(measures):
        grid = measure_grids[idx]
        for event in measure.get("events", []):
            start_tick = int(event.get("startTick", 0))
            if start_tick < 0 or start_tick >= measure_ticks:
                warning = f"Tic de départ hors mesure {idx + 1}: {start_tick}"
                warnings.append(warning)
                if logger:
                    logger.warning(warning)
                continue
            for note in event.get("notes", []):
                string_number = int(note.get("string", 1))
                if string_number < 1 or string_number > 6:
                    warning = f"Corde invalide ({string_number}) dans la mesure {idx + 1}."
                    warnings.append(warning)
                    if logger:
                        logger.warning(warning)
                    continue
                row = grid[string_number]
                fret_str = str(note.get("fret", ""))
                token = (fret_str + "-" * (3 - len(fret_str)))[:3]
                if row[start_tick] != "---":
                    warning = (
                        f"Collision mesure {idx + 1}, corde {string_number}, tick {start_tick}: "
                        f"{row[start_tick]} déjà présent, {token} ignoré."
                    )
                    warnings.append(warning)
                    if logger:
                        logger.warning(warning)
                    continue
                row[start_tick] = token
                note_count += 1

    tuning_names = metadata.get("tuning") or []
    tempo_value = metadata.get("tempo")
    header_lines: List[str] = []
    if tuning_names:
        header_lines.append(f"Accordage: {', '.join(str(name) for name in tuning_names)}")
    if tempo_value is not None:
        tempo_str = f"{int(round(tempo_value))}" if isinstance(tempo_value, (int, float)) else str(tempo_value)
        header_lines.append(f"Tempo: {tempo_str}")
    if quality:
        header_lines.append(f"Qualité: {quality}")
    if header_lines:
        header_lines.append("")

    line_order = [1, 2, 3, 4, 5, 6]
    names = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}
    lines: List[str] = []
    for block_start in range(0, measure_count, measures_per_system):
        block_grids = measure_grids[block_start : block_start + measures_per_system]
        if not block_grids:
            continue
        block_label = f"Mesures {block_start + 1}-{block_start + len(block_grids)}"
        lines.append(block_label)
        for string in line_order:
            block_cells = "|".join("".join(grid[string]) for grid in block_grids)
            lines.append(f"{names[string]}|{block_cells}|")
        lines.append("")

    output_lines = header_lines + lines if header_lines else lines
    while output_lines and output_lines[-1] == "":
        output_lines.pop()

    return "\n".join(output_lines), note_count, warnings


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
    logger=None,
    measures_per_system: int = 2,
    pretty_export: bool = False,
) -> Tuple[Dict[str, object], int, List[str]]:
    tab_json, total_notes = build_tab_json(tab_notes, tuning, capo, quality, tempo_bpm, tempo_source, warnings)
    if total_notes == 0:
        raise RuntimeError("Aucune tablature générée (pas de notes exploitables).")
    write_json(tab_json_path, tab_json)
    tab_txt, tab_txt_notes, render_warnings = render_tab_txt_from_json(
        tab_json, quality, measures_per_system=measures_per_system, logger=logger
    )
    with open(tab_txt_path, "w", encoding="utf-8") as f:
        f.write(tab_txt)

    if pretty_export:
        pretty_path = os.path.join(os.path.dirname(tab_txt_path), "tab.pretty.txt")
        pretty_text = tab_txt.replace("|", " | ").replace("---", "   ")
        with open(pretty_path, "w", encoding="utf-8") as pretty_file:
            pretty_file.write(pretty_text)

    output_warnings = list(render_warnings)
    if total_notes != tab_txt_notes:
        diff_path = os.path.join(os.path.dirname(tab_txt_path), "diff_tabtxt.txt")
        with open(diff_path, "w", encoding="utf-8") as diff_file:
            diff_file.write(f"Notes tab.json : {total_notes}\n")
            diff_file.write(f"Notes tab.txt : {tab_txt_notes}\n")
        warning = (
            f"Différence entre tab.txt ({tab_txt_notes}) et tab.json ({total_notes}) "
            f"(voir {os.path.basename(diff_path)})."
        )
        output_warnings.append(warning)
        if logger:
            logger.warning(warning)

    return tab_json, tab_txt_notes, output_warnings


def load_tab_json(path: str) -> Dict[str, object]:
    return read_json(path)


def tab_json_note_tuples(tab_json: Dict[str, Any]) -> List[Tuple[Optional[int], int, int]]:
    tuples: List[Tuple[Optional[int], int, int]] = []
    for measure in tab_json.get("measures", []):
        measure_number = measure.get("number")
        for event in measure.get("events", []):
            for note in event.get("notes", []):
                string_number = int(note.get("string", 0))
                fret_number = int(note.get("fret", 0))
                tuples.append(
                    (
                        int(measure_number) if isinstance(measure_number, int) else None,
                        string_number,
                        fret_number,
                    )
                )
    return tuples


def parse_musicxml_notes(xml_path: str) -> List[Tuple[Optional[int], int, int]]:
    if not os.path.exists(xml_path):
        return []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    part = root.find("part")
    if part is None:
        return []
    collected: List[Tuple[Optional[int], int, int]] = []
    for measure in part.findall("measure"):
        measure_number = measure.get("number")
        for note_el in measure.findall("note"):
            technical = note_el.find("notations/technical")
            if technical is None:
                continue
            string_text = technical.findtext("string")
            fret_text = technical.findtext("fret")
            if string_text is None or fret_text is None:
                continue
            try:
                string_num = int(string_text)
                fret_num = int(fret_text)
            except ValueError:
                continue
            collected.append(
                (
                    int(measure_number) if measure_number and measure_number.isdigit() else None,
                    string_num,
                    fret_num,
                )
            )
    return collected


def compare_tab_json_and_musicxml(
    tab_json: Dict[str, Any],
    xml_path: str,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    tab_tuples = tab_json_note_tuples(tab_json)
    xml_tuples = parse_musicxml_notes(xml_path)
    tab_counter = Counter(tab_tuples)
    xml_counter = Counter(xml_tuples)
    diffs: List[Dict[str, Any]] = []
    for key in set(tab_counter) | set(xml_counter):
        tab_count = tab_counter.get(key, 0)
        xml_count = xml_counter.get(key, 0)
        if tab_count != xml_count:
            diffs.append(
                {
                    "measure": key[0],
                    "string": key[1],
                    "fret": key[2],
                    "countTabJson": tab_count,
                    "countMusicXML": xml_count,
                }
            )
    return len(tab_tuples), len(xml_tuples), diffs


def _midi_to_pitch(midi: int) -> Tuple[str, int, int]:
    steps = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
    alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
    step = steps[midi % 12]
    alter = alters[midi % 12]
    octave = midi // 12 - 1
    return step, alter, octave


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
    tab_json: Dict[str, object],
    tuning: str,
    capo: int,
    tempo_bpm: float,
    title: str,
    artist: str,
    annotations: Optional[List[Dict[str, str]]],
    output_path: str,
    logger=None,
) -> Tuple[int, int, List[Dict[str, Any]]]:
    metadata = tab_json.get("metadata", {})
    divisions = int(metadata.get("divisions", 4) or 4)
    seconds_per_beat = 60.0 / tempo_bpm if tempo_bpm else 0.0
    open_strings, _ = parse_tuning(tuning, capo)
    measures = tab_json.get("measures", [])
    measure_count = max(len(measures), 1)

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

        measure_data = measures[measure_idx] if measure_idx < len(measures) else {"events": []}
        measure_events = list(measure_data.get("events", []))
        items, beam_status = _build_measure_items(measure_events, divisions)

        for item_idx, item in enumerate(items):
            duration = int(item["duration"])
            if item["kind"] == "rest":
                note_el = ET.SubElement(measure, "note")
                ET.SubElement(note_el, "rest")
                ET.SubElement(note_el, "duration").text = str(duration)
                ET.SubElement(note_el, "type").text = _duration_type(duration, divisions)
                continue

            notes_data = item.get("notes", [])
            base_tie_start = bool(item.get("tie_start"))
            base_tie_stop = bool(item.get("tie_stop"))
            for n_idx, chord_note in enumerate(notes_data):
                note_el = ET.SubElement(measure, "note")
                if n_idx > 0:
                    ET.SubElement(note_el, "chord")
                string_number = int(chord_note.get("string", 1))
                fret_number = int(chord_note.get("fret", 0))
                midi_value = pitch_from_string_fret(string_number, fret_number, open_strings)
                step, alter, octave = _midi_to_pitch(midi_value)
                pitch_el = ET.SubElement(note_el, "pitch")
                ET.SubElement(pitch_el, "step").text = step
                if alter:
                    ET.SubElement(pitch_el, "alter").text = str(alter)
                ET.SubElement(pitch_el, "octave").text = str(octave)
                ET.SubElement(note_el, "duration").text = str(duration)
                ET.SubElement(note_el, "type").text = _duration_type(duration, divisions)
                ET.SubElement(note_el, "voice").text = "1"
                if item_idx in beam_status and n_idx == 0:
                    ET.SubElement(note_el, "beam", number="1").text = beam_status[item_idx]
                notations = ET.SubElement(note_el, "notations")
                if base_tie_start:
                    ET.SubElement(notations, "tied", type="start")
                    ET.SubElement(note_el, "tie", type="start")
                if base_tie_stop:
                    ET.SubElement(notations, "tied", type="stop")
                    ET.SubElement(note_el, "tie", type="stop")
                technical = ET.SubElement(notations, "technical")
                ET.SubElement(technical, "string").text = str(string_number)
                ET.SubElement(technical, "fret").text = str(fret_number)

    _indent_xml(score)
    tree = ET.ElementTree(score)
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
            b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
        )
        tree.write(f, encoding="utf-8")
    tab_count, xml_count, diff_report = compare_tab_json_and_musicxml(tab_json, output_path)
    if diff_report:
        warning = (
            "Mismatch entre tab.json et MusicXML: "
            f"{len(diff_report)} différences (exemple {diff_report[:3]})."
        )
        if logger:
            logger.warning(warning)
        else:
            print(warning)
    return tab_count, xml_count, diff_report


def write_score_musicxml(
    score_json: Dict[str, Any],
    title: str,
    artist: str,
    annotations: Optional[List[Dict[str, str]]],
    output_path: str,
    logger=None,
) -> Tuple[int, int]:
    metadata = score_json.get("metadata", {})
    divisions = int(metadata.get("divisions", DEFAULT_DIVISIONS) or DEFAULT_DIVISIONS)
    tempo_bpm = float(metadata.get("tempo") or 0)
    time_sig = metadata.get("timeSignature", DEFAULT_TIME_SIGNATURE)
    written_shift = int(metadata.get("scoreWrittenOctaveShift", GUITAR_WRITTEN_SHIFT))
    try:
        beats, beat_type = [int(part) for part in time_sig.split("/") if part]
    except ValueError:
        beats, beat_type = 4, 4
    annotations = annotations or []
    seconds_per_beat = _seconds_per_beat(tempo_bpm)
    annotations_by_measure: Dict[int, List[Dict[str, str]]] = {}
    for item in annotations:
        try:
            ts = float(item.get("timeStart", 0))
        except (TypeError, ValueError):
            ts = 0.0
        idx = int((ts / seconds_per_beat) // beats) if seconds_per_beat else 0
        annotations_by_measure.setdefault(idx, []).append(item)

    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title or "Partition"
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
    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Partition guitare"

    part = ET.SubElement(score, "part", id="P1")
    section_labels = ["A", "B", "C", "D", "E"]
    measures = score_json.get("measures", [])
    measure_count = max(len(measures), 1)
    transpose_value = -written_shift
    score_note_count = sum(len(event["notes"]) for measure in measures for event in measure["events"])
    xml_note_count = 0

    for measure_idx in range(measure_count):
        measure = ET.SubElement(part, "measure", number=str(measure_idx + 1))
        if measure_idx == 0:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(divisions)
            key = ET.SubElement(attributes, "key")
            ET.SubElement(key, "fifths").text = "0"
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = str(beats)
            ET.SubElement(time, "beat-type").text = str(beat_type)
            transpose = ET.SubElement(attributes, "transpose")
            ET.SubElement(transpose, "chromatic").text = str(transpose_value)
            ET.SubElement(transpose, "diatonic").text = "0"
            clef = ET.SubElement(attributes, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"
            ET.SubElement(clef, "clef-octave-change").text = "-1"
            staff_details = ET.SubElement(attributes, "staff-details")
            ET.SubElement(staff_details, "staff-lines").text = "5"
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

        measure_data = measures[measure_idx] if measure_idx < len(measures) else {"events": []}
        measure_events = list(measure_data.get("events", []))
        items, beam_status = _build_measure_items(measure_events, divisions)
        for item_idx, item in enumerate(items):
            duration = int(item["duration"])
            if item["kind"] == "rest":
                note_el = ET.SubElement(measure, "note")
                ET.SubElement(note_el, "rest")
                ET.SubElement(note_el, "duration").text = str(duration)
                ET.SubElement(note_el, "type").text = _duration_type(duration, divisions)
                continue

            notes_data = item.get("notes", [])
            base_tie_start = bool(item.get("tie_start"))
            base_tie_stop = bool(item.get("tie_stop"))
            for n_idx, note_data in enumerate(notes_data):
                note_el = ET.SubElement(measure, "note")
                if n_idx > 0:
                    ET.SubElement(note_el, "chord")
                midi_value = int(note_data.get("midiPitch", 0))
                step, alter, octave = _midi_to_pitch(midi_value)
                pitch_el = ET.SubElement(note_el, "pitch")
                ET.SubElement(pitch_el, "step").text = step
                if alter:
                    ET.SubElement(pitch_el, "alter").text = str(alter)
                ET.SubElement(pitch_el, "octave").text = str(octave)
                ET.SubElement(note_el, "duration").text = str(duration)
                ET.SubElement(note_el, "type").text = _duration_type(duration, divisions)
                ET.SubElement(note_el, "voice").text = "1"
                if item_idx in beam_status and n_idx == 0:
                    ET.SubElement(note_el, "beam", number="1").text = beam_status[item_idx]
                notations = ET.SubElement(note_el, "notations")
                if base_tie_start:
                    ET.SubElement(notations, "tied", type="start")
                    ET.SubElement(note_el, "tie", type="start")
                if base_tie_stop:
                    ET.SubElement(notations, "tied", type="stop")
                    ET.SubElement(note_el, "tie", type="stop")
                xml_note_count += 1

    _indent_xml(score)
    tree = ET.ElementTree(score)
    with open(output_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
            b'"http://www.musicxml.org/dtds/partwise.dtd">\n'
        )
        tree.write(f, encoding="utf-8")

    if score_note_count != xml_note_count:
        message = f"Mismatch entre score.json ({score_note_count}) et MusicXML ({xml_note_count})."
        if logger:
            logger.error(message)
        else:
            print(message)
        raise RuntimeError(message)
    return score_note_count, xml_note_count


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

