package com.tabscore.app.domain.model

import java.util.UUID

data class Transcription(
    val id: UUID,
    val title: String,
    val sourceType: SourceType,
    val sourceUri: String,
    val outputType: OutputType,
    val instrument: String,
    val tuning: GuitarTuning,
    val capo: Int,
    val mode: GuitarMode,
    val createdAt: Long,
    val status: TranscriptionStatus,
    val progress: Int,
    val stage: String?,
    val resultMusicXmlPath: String?,
    val resultTabPath: String?,
    val resultTabJsonPath: String?,
    val resultMidiPath: String?,
    val startSeconds: Int?,
    val endSeconds: Int?,
    val errorMessage: String?,
    val durationSeconds: Int?,
    val confidence: Float?,
    val jobId: String?
)

enum class SourceType {
    AUDIO_FILE,
    YOUTUBE_URL
}

enum class OutputType {
    SCORE,
    TAB,
    BOTH
}

enum class TranscriptionStatus {
    PENDING,
    RUNNING,
    DONE,
    FAILED
}

enum class Quality {
    FAST,
    ACCURATE
}

enum class GuitarTuning(val apiValue: String, val displayName: String) {
    STANDARD("EADGBE", "Standard EADGBE"),
    DROP_D("DADGBE", "Drop D"),
    OPEN_G("DGDGBD", "Open G")
}

enum class GuitarMode(val displayName: String, val apiValue: String) {
    BEST_EFFORT("Guitare (best effort)", "best_effort"),
    ISOLATED_TRACK("Guitare (piste isolée)", "isolated_track")
}

enum class ExportFormat {
    PDF,
    MUSICXML,
    TXT
}
