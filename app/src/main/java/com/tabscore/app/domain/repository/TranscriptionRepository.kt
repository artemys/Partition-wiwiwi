package com.tabscore.app.domain.repository

import com.tabscore.app.domain.model.OutputType
import com.tabscore.app.domain.model.Quality
import com.tabscore.app.domain.model.Transcription
import kotlinx.coroutines.flow.Flow
import java.util.UUID

interface TranscriptionRepository {
    fun observeAll(): Flow<List<Transcription>>
    fun observeById(id: UUID): Flow<Transcription?>
    suspend fun createFromAudio(
        audioUri: String,
        title: String,
        outputType: OutputType,
        instrument: String,
        tuning: com.tabscore.app.domain.model.GuitarTuning,
        capo: Int,
        mode: com.tabscore.app.domain.model.GuitarMode,
        quality: Quality,
        startSeconds: Int?,
        endSeconds: Int?
    ): UUID

    suspend fun createFromYoutube(
        youtubeUrl: String,
        title: String,
        outputType: OutputType,
        instrument: String,
        tuning: com.tabscore.app.domain.model.GuitarTuning,
        capo: Int,
        mode: com.tabscore.app.domain.model.GuitarMode,
        quality: Quality,
        startSeconds: Int?,
        endSeconds: Int?
    ): UUID

    suspend fun delete(id: UUID)
    suspend fun retry(id: UUID)
    suspend fun downloadExports(id: UUID): List<String>
}
