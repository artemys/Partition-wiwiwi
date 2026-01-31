package com.tabscore.app.data.mapper

import com.tabscore.app.data.local.TranscriptionEntity
import com.tabscore.app.domain.model.Transcription

fun TranscriptionEntity.toDomain(): Transcription = Transcription(
    id = id,
    title = title,
    sourceType = sourceType,
    sourceUri = sourceUri,
    outputType = outputType,
    instrument = instrument,
    tuning = tuning,
    capo = capo,
    mode = mode,
    createdAt = createdAt,
    status = status,
    progress = progress,
    stage = stage,
    resultMusicXmlPath = resultMusicXmlPath,
    resultTabPath = resultTabPath,
    resultTabJsonPath = resultTabJsonPath,
    resultMidiPath = resultMidiPath,
    startSeconds = startSeconds,
    endSeconds = endSeconds,
    errorMessage = errorMessage,
    durationSeconds = durationSeconds,
    confidence = confidence,
    jobId = jobId
)

fun Transcription.toEntity(): TranscriptionEntity = TranscriptionEntity(
    id = id,
    title = title,
    sourceType = sourceType,
    sourceUri = sourceUri,
    outputType = outputType,
    instrument = instrument,
    tuning = tuning,
    capo = capo,
    mode = mode,
    createdAt = createdAt,
    status = status,
    progress = progress,
    stage = stage,
    resultMusicXmlPath = resultMusicXmlPath,
    resultTabPath = resultTabPath,
    resultTabJsonPath = resultTabJsonPath,
    resultMidiPath = resultMidiPath,
    startSeconds = startSeconds,
    endSeconds = endSeconds,
    errorMessage = errorMessage,
    durationSeconds = durationSeconds,
    confidence = confidence,
    jobId = jobId
)
