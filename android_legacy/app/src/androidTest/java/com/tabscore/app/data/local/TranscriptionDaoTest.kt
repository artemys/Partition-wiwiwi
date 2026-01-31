package com.tabscore.app.data.local

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.tabscore.app.domain.model.GuitarMode
import com.tabscore.app.domain.model.GuitarTuning
import com.tabscore.app.domain.model.OutputType
import com.tabscore.app.domain.model.SourceType
import com.tabscore.app.domain.model.TranscriptionStatus
import com.google.common.truth.Truth.assertThat
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class TranscriptionDaoTest {

    private lateinit var db: TabScoreDatabase
    private lateinit var dao: TranscriptionDao

    @Before
    fun setup() {
        db = Room.inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext(),
            TabScoreDatabase::class.java
        ).allowMainThreadQueries().build()
        dao = db.transcriptionDao()
    }

    @After
    fun tearDown() {
        db.close()
    }

    @Test
    fun upsertAndObserve() = runBlocking {
        val id = UUID.randomUUID()
        val entity = TranscriptionEntity(
            id = id,
            title = "Test",
            sourceType = SourceType.AUDIO_FILE,
            sourceUri = "content://audio",
            outputType = OutputType.SCORE,
            instrument = "Guitare",
            tuning = GuitarTuning.STANDARD,
            capo = 0,
            mode = GuitarMode.BEST_EFFORT,
            createdAt = 123L,
            status = TranscriptionStatus.PENDING,
            resultMusicXmlPath = null,
            resultTabPath = null,
            errorMessage = null,
            durationSeconds = 120,
            jobId = "job-1"
        )
        dao.upsert(entity)
        val all = dao.observeAll().first()
        assertThat(all).hasSize(1)
        assertThat(all.first().id).isEqualTo(id)
    }
}
