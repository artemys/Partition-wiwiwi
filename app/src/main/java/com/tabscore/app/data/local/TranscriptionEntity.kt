package com.tabscore.app.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey
import com.tabscore.app.domain.model.OutputType
import com.tabscore.app.domain.model.SourceType
import com.tabscore.app.domain.model.TranscriptionStatus
import java.util.UUID

@Entity(tableName = "transcriptions")
data class TranscriptionEntity(
    @PrimaryKey val id: UUID,
    val title: String,
    val sourceType: SourceType,
    val sourceUri: String,
    val outputType: OutputType,
    val instrument: String,
    val tuning: com.tabscore.app.domain.model.GuitarTuning,
    val capo: Int,
    val mode: com.tabscore.app.domain.model.GuitarMode,
    val createdAt: Long,
    val status: TranscriptionStatus,
    val progress: Int,
    val stage: String?,
    val resultMusicXmlPath: String?,
    val resultTabPath: String?,
    val resultTabJsonPath: String?,
    val resultMidiPath: String?,
    val errorMessage: String?,
    val durationSeconds: Int?,
    val confidence: Float?,
    val jobId: String?
)
