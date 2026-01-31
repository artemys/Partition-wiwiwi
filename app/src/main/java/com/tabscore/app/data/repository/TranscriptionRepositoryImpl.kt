package com.tabscore.app.data.repository

import com.tabscore.app.core.di.ApplicationScope
import com.tabscore.app.data.local.TranscriptionDao
import com.tabscore.app.data.local.TranscriptionEntity
import com.tabscore.app.data.mapper.toDomain
import com.tabscore.app.data.mapper.toEntity
import com.tabscore.app.data.remote.FileDownloader
import com.tabscore.app.data.remote.TabScoreRemoteApi
import com.tabscore.app.data.remote.model.JobStatus
import com.tabscore.app.data.settings.SettingsRepository
import com.tabscore.app.data.storage.AudioReader
import com.tabscore.app.data.storage.ExportStorage
import com.tabscore.app.domain.model.OutputType
import com.tabscore.app.domain.model.Quality
import com.tabscore.app.domain.model.SourceType
import com.tabscore.app.domain.model.Transcription
import com.tabscore.app.domain.model.TranscriptionStatus
import com.tabscore.app.domain.repository.TranscriptionRepository
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.flow.first
import java.util.UUID

class TranscriptionRepositoryImpl(
    private val dao: TranscriptionDao,
    private val remoteApi: TabScoreRemoteApi,
    private val fileDownloader: FileDownloader,
    private val exportStorage: ExportStorage,
    private val audioReader: AudioReader,
    private val settingsRepository: SettingsRepository,
    @ApplicationScope private val appScope: CoroutineScope,
    private val ioDispatcher: CoroutineDispatcher
) : TranscriptionRepository {

    override fun observeAll(): Flow<List<Transcription>> =
        dao.observeAll().map { list -> list.map { it.toDomain() } }

    override fun observeById(id: UUID): Flow<Transcription?> =
        dao.observeById(id).map { it?.toDomain() }

    override suspend fun createFromAudio(
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
    ): UUID = withContext(ioDispatcher) {
        val id = UUID.randomUUID()
        val entity = TranscriptionEntity(
            id = id,
            title = title,
            sourceType = SourceType.AUDIO_FILE,
            sourceUri = audioUri,
            outputType = outputType,
            instrument = instrument,
            tuning = tuning,
            capo = capo,
            mode = mode,
            createdAt = System.currentTimeMillis(),
            status = TranscriptionStatus.PENDING,
            progress = 0,
            stage = "PENDING",
            resultMusicXmlPath = null,
            resultTabPath = null,
            resultTabJsonPath = null,
            resultMidiPath = null,
            startSeconds = startSeconds,
            endSeconds = endSeconds,
            errorMessage = null,
            durationSeconds = null,
            confidence = null,
            jobId = null
        )
        dao.upsert(entity)
        val payload = audioReader.readBytes(audioUri)
        val response = remoteApi.createJobFromAudio(
            audioBytes = payload.bytes,
            filename = payload.filename,
            mimeType = payload.mimeType,
            targetInstrument = "guitar",
            target = "GUITAR_BEST_EFFORT",
            outputType = outputType.name.lowercase(),
            tuning = tuning.apiValue,
            capo = capo,
            mode = mode.apiValue,
            quality = quality.name.lowercase(),
            startSeconds = startSeconds,
            endSeconds = endSeconds
        )
        dao.upsert(entity.copy(status = TranscriptionStatus.RUNNING, progress = 5, stage = "PENDING", jobId = response.jobId))
        startPolling(id, response.jobId)
        id
    }

    override suspend fun createFromYoutube(
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
    ): UUID = withContext(ioDispatcher) {
        val id = UUID.randomUUID()
        val entity = TranscriptionEntity(
            id = id,
            title = title,
            sourceType = SourceType.YOUTUBE_URL,
            sourceUri = youtubeUrl,
            outputType = outputType,
            instrument = instrument,
            tuning = tuning,
            capo = capo,
            mode = mode,
            createdAt = System.currentTimeMillis(),
            status = TranscriptionStatus.PENDING,
            progress = 0,
            stage = "PENDING",
            resultMusicXmlPath = null,
            resultTabPath = null,
            resultTabJsonPath = null,
            resultMidiPath = null,
            startSeconds = startSeconds,
            endSeconds = endSeconds,
            errorMessage = null,
            durationSeconds = null,
            confidence = null,
            jobId = null
        )
        dao.upsert(entity)
        val response = remoteApi.createJobFromYoutube(
            youtubeUrl = youtubeUrl,
            targetInstrument = "guitar",
            target = "GUITAR_BEST_EFFORT",
            outputType = outputType.name.lowercase(),
            tuning = tuning.apiValue,
            capo = capo,
            mode = mode.apiValue,
            quality = quality.name.lowercase(),
            startSeconds = startSeconds,
            endSeconds = endSeconds
        )
        dao.upsert(entity.copy(status = TranscriptionStatus.RUNNING, progress = 5, stage = "PENDING", jobId = response.jobId))
        startPolling(id, response.jobId)
        id
    }

    override suspend fun delete(id: UUID) {
        dao.getById(id)?.let { entity ->
            entity.jobId?.let { remoteApi.deleteJob(it) }
            dao.delete(entity)
        }
    }

    override suspend fun retry(id: UUID) {
        val entity = dao.getById(id) ?: return
        val updated = entity.copy(
            status = TranscriptionStatus.PENDING,
            progress = 0,
            stage = "PENDING",
            errorMessage = null,
            resultMusicXmlPath = null,
            resultTabPath = null,
            resultTabJsonPath = null,
            resultMidiPath = null,
            confidence = null
        )
        dao.upsert(updated)
        val response = if (entity.sourceType == SourceType.AUDIO_FILE) {
            val payload = audioReader.readBytes(entity.sourceUri)
            remoteApi.createJobFromAudio(
                audioBytes = payload.bytes,
                filename = payload.filename,
                mimeType = payload.mimeType,
                targetInstrument = "guitar",
                target = "GUITAR_BEST_EFFORT",
                outputType = entity.outputType.name.lowercase(),
                tuning = entity.tuning.apiValue,
                capo = entity.capo,
                mode = entity.mode.apiValue,
                quality = Quality.ACCURATE.name.lowercase(),
                startSeconds = entity.startSeconds,
                endSeconds = entity.endSeconds
            )
        } else {
            remoteApi.createJobFromYoutube(
                youtubeUrl = entity.sourceUri,
                targetInstrument = "guitar",
                target = "GUITAR_BEST_EFFORT",
                outputType = entity.outputType.name.lowercase(),
                tuning = entity.tuning.apiValue,
                capo = entity.capo,
                mode = entity.mode.apiValue,
                quality = Quality.ACCURATE.name.lowercase(),
                startSeconds = entity.startSeconds,
                endSeconds = entity.endSeconds
            )
        }
        dao.upsert(updated.copy(status = TranscriptionStatus.RUNNING, progress = 5, stage = "PENDING", jobId = response.jobId))
        startPolling(id, response.jobId)
    }

    override suspend fun downloadExports(id: UUID): List<String> = withContext(ioDispatcher) {
        val entity = dao.getById(id) ?: return@withContext emptyList()
        val existing = listOfNotNull(
            entity.resultMusicXmlPath,
            entity.resultTabPath,
            entity.resultTabJsonPath,
            entity.resultMidiPath
        )
        if (existing.isNotEmpty()) {
            return@withContext existing
        }
        val jobId = entity.jobId ?: return@withContext emptyList()
        val jobResult = remoteApi.getJobResult(jobId)
        val musicXmlPath = jobResult.musicXmlUrl?.let { url ->
            val bytes = fileDownloader.download(url)
            exportStorage.saveBytes("tabscore_${id}_score.xml", bytes)
        }
        val tabPath = jobResult.tabTxtUrl?.let { url ->
            val bytes = fileDownloader.download(url)
            exportStorage.saveBytes("tabscore_${id}_tab.txt", bytes)
        }
        val tabJsonPath = jobResult.tabJsonUrl?.let { url ->
            val bytes = fileDownloader.download(url)
            exportStorage.saveBytes("tabscore_${id}_tab.json", bytes)
        }
        val midiPath = jobResult.midiUrl?.let { url ->
            val bytes = fileDownloader.download(url)
            exportStorage.saveBytes("tabscore_${id}_output.mid", bytes)
        }
        val updated = entity.copy(
            resultMusicXmlPath = musicXmlPath ?: entity.resultMusicXmlPath,
            resultTabPath = tabPath ?: entity.resultTabPath,
            resultTabJsonPath = tabJsonPath ?: entity.resultTabJsonPath,
            resultMidiPath = midiPath ?: entity.resultMidiPath,
            durationSeconds = jobResult.metadata?.durationSeconds?.toInt() ?: entity.durationSeconds
        )
        dao.upsert(updated)
        listOfNotNull(musicXmlPath, tabPath, tabJsonPath, midiPath)
    }

    private fun startPolling(transcriptionId: UUID, jobId: String) {
        appScope.launch {
            var finished = false
            var delayMs = 2000L
            while (!finished) {
                val status = remoteApi.getJobStatus(jobId)
                when (status.status) {
                    JobStatus.PENDING, JobStatus.RUNNING -> {
                        dao.getById(transcriptionId)?.let { current ->
                            dao.upsert(
                                current.copy(
                                    status = TranscriptionStatus.RUNNING,
                                    progress = status.progress,
                                    stage = status.stage ?: current.stage,
                                    confidence = status.confidence
                                )
                            )
                        }
                    }
                    JobStatus.DONE -> {
                        val result = remoteApi.getJobResult(jobId)
                        val musicXmlPath = result.musicXmlUrl?.let { url ->
                            val bytes = fileDownloader.download(url)
                            exportStorage.saveBytes("tabscore_${transcriptionId}_score.xml", bytes)
                        }
                        val tabPath = result.tabTxtUrl?.let { url ->
                            val bytes = fileDownloader.download(url)
                            exportStorage.saveBytes("tabscore_${transcriptionId}_tab.txt", bytes)
                        }
                        val tabJsonPath = result.tabJsonUrl?.let { url ->
                            val bytes = fileDownloader.download(url)
                            exportStorage.saveBytes("tabscore_${transcriptionId}_tab.json", bytes)
                        }
                        val midiPath = result.midiUrl?.let { url ->
                            val bytes = fileDownloader.download(url)
                            exportStorage.saveBytes("tabscore_${transcriptionId}_output.mid", bytes)
                        }
                        dao.getById(transcriptionId)?.let { current ->
                            dao.upsert(
                                current.copy(
                                    status = TranscriptionStatus.DONE,
                                    progress = 100,
                                    stage = "DONE",
                                    resultMusicXmlPath = musicXmlPath,
                                    resultTabPath = tabPath,
                                    resultTabJsonPath = tabJsonPath,
                                    resultMidiPath = midiPath,
                                    durationSeconds = result.metadata?.durationSeconds?.toInt(),
                                    errorMessage = null,
                                    confidence = current.confidence ?: status.confidence
                                )
                            )
                        }
                        finished = true
                    }
                    JobStatus.FAILED -> {
                        dao.getById(transcriptionId)?.let { current ->
                            dao.upsert(
                                current.copy(
                                    status = TranscriptionStatus.FAILED,
                                    progress = status.progress,
                                    stage = status.stage ?: "FAILED",
                                    errorMessage = status.errorMessage ?: "Échec du traitement"
                                )
                            )
                        }
                        finished = true
                    }
                }
                if (!finished) {
                    delay(delayMs)
                    if (delayMs < 5000L) {
                        delayMs = 5000L
                    }
                }
            }
        }
    }
}
