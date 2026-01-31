package com.tabscore.app.data.repository

import com.tabscore.app.data.local.TranscriptionDao
import com.tabscore.app.data.local.TranscriptionEntity
import com.tabscore.app.data.remote.FileDownloader
import com.tabscore.app.data.remote.FakeTabScoreApi
import com.tabscore.app.data.remote.TabScoreRemoteApi
import com.tabscore.app.data.settings.SettingsRepository
import com.tabscore.app.data.storage.AudioPayload
import com.tabscore.app.data.storage.AudioReader
import com.tabscore.app.data.storage.ExportStorage
import com.tabscore.app.domain.model.GuitarMode
import com.tabscore.app.domain.model.GuitarTuning
import com.tabscore.app.domain.model.OutputType
import com.tabscore.app.domain.model.Quality
import com.tabscore.app.domain.model.SourceType
import com.tabscore.app.domain.model.TranscriptionStatus
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.runTest
import org.junit.Test
import java.util.UUID

class TranscriptionRepositoryImplTest {

    @Test
    fun createFromYoutube_completesAndSavesResults() = runTest {
        val dispatcher = StandardTestDispatcher(testScheduler)
        val scope = TestScope(dispatcher)

        val dao = FakeTranscriptionDao()
        val api: TabScoreRemoteApi = FakeTabScoreApi()
        val fileDownloader = FakeFileDownloader()
        val storage = FakeExportStorage()
        val audioReader = FakeAudioReader()
        val settings = FakeSettingsRepository()

        val repository = TranscriptionRepositoryImpl(
            dao = dao,
            remoteApi = api,
            fileDownloader = fileDownloader,
            exportStorage = storage,
            audioReader = audioReader,
            settingsRepository = settings,
            appScope = scope,
            ioDispatcher = dispatcher
        )

        val deferred = async {
            repository.createFromYoutube(
                youtubeUrl = "https://youtube.com/watch?v=123",
                title = "Test",
                outputType = OutputType.SCORE,
                instrument = "Guitare",
                tuning = GuitarTuning.STANDARD,
                capo = 0,
                mode = GuitarMode.BEST_EFFORT,
                quality = Quality.ACCURATE
            )
        }
        advanceTimeBy(1_000)
        val id = deferred.await()

        advanceTimeBy(9_000)
        val transcription = dao.getById(id)
        assertThat(transcription?.status).isEqualTo(TranscriptionStatus.DONE)
        assertThat(transcription?.resultMusicXmlPath).isNotNull()
        assertThat(storage.saved.isNotEmpty()).isTrue()
    }
}

private class FakeTranscriptionDao : TranscriptionDao {
    private val state = MutableStateFlow<List<TranscriptionEntity>>(emptyList())

    override fun observeAll(): Flow<List<TranscriptionEntity>> = state

    override fun observeById(id: UUID): Flow<TranscriptionEntity?> =
        state.map { list -> list.find { it.id == id } }

    override suspend fun upsert(entity: TranscriptionEntity) {
        val list = state.value.toMutableList()
        val index = list.indexOfFirst { it.id == entity.id }
        if (index >= 0) list[index] = entity else list.add(entity)
        state.value = list
    }

    override suspend fun update(entity: TranscriptionEntity) = upsert(entity)

    override suspend fun delete(entity: TranscriptionEntity) {
        state.value = state.value.filterNot { it.id == entity.id }
    }

    override suspend fun getById(id: UUID): TranscriptionEntity? =
        state.value.find { it.id == id }
}

private class FakeFileDownloader : FileDownloader {
    override suspend fun download(url: String): ByteArray = "dummy".toByteArray()
}

private class FakeExportStorage : ExportStorage {
    val saved = mutableMapOf<String, ByteArray>()
    override suspend fun saveBytes(filename: String, bytes: ByteArray): String {
        saved[filename] = bytes
        return "/tmp/$filename"
    }
}

private class FakeAudioReader : AudioReader {
    override suspend fun readBytes(uriString: String): AudioPayload =
        AudioPayload("audio".toByteArray(), "audio.mp3", "audio/mpeg")
}

private class FakeSettingsRepository : SettingsRepository {
    override val exportFormat = MutableStateFlow(com.tabscore.app.domain.model.ExportFormat.MUSICXML)
    override val quality = MutableStateFlow(Quality.ACCURATE)
    override suspend fun setExportFormat(format: com.tabscore.app.domain.model.ExportFormat) {
        exportFormat.value = format
    }

    override suspend fun setQuality(quality: Quality) {
        this.quality.value = quality
    }
}
