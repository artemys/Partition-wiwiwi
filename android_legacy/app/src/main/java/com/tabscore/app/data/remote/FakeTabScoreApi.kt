package com.tabscore.app.data.remote

import com.tabscore.app.data.remote.model.CreateJobResponse
import com.tabscore.app.data.remote.model.JobMetadata
import com.tabscore.app.data.remote.model.JobResultResponse
import com.tabscore.app.data.remote.model.JobStatus
import com.tabscore.app.data.remote.model.JobStatusResponse
import kotlinx.coroutines.delay
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

class FakeTabScoreApi : TabScoreRemoteApi {

    private data class JobEntry(
        val id: String,
        val title: String,
        val createdAt: Long,
        val outputType: String,
        val durationSeconds: Int
    )

    private val jobs = ConcurrentHashMap<String, JobEntry>()

    override suspend fun createJobFromAudio(
        audioBytes: ByteArray,
        filename: String,
        mimeType: String,
        targetInstrument: String,
        target: String,
        outputType: String,
        tuning: String,
        capo: Int,
        mode: String,
        quality: String,
        startSeconds: Int?,
        endSeconds: Int?
    ): CreateJobResponse {
        delay(400)
        val jobId = UUID.randomUUID().toString()
        jobs[jobId] = JobEntry(
            id = jobId,
            title = filename,
            createdAt = System.currentTimeMillis(),
            outputType = outputType,
            durationSeconds = 132
        )
        return CreateJobResponse(jobId = jobId, status = "PENDING")
    }

    override suspend fun createJobFromYoutube(
        youtubeUrl: String,
        targetInstrument: String,
        target: String,
        outputType: String,
        tuning: String,
        capo: Int,
        mode: String,
        quality: String,
        startSeconds: Int?,
        endSeconds: Int?
    ): CreateJobResponse {
        delay(400)
        val jobId = UUID.randomUUID().toString()
        jobs[jobId] = JobEntry(
            id = jobId,
            title = "YouTube",
            createdAt = System.currentTimeMillis(),
            outputType = outputType,
            durationSeconds = 158
        )
        return CreateJobResponse(jobId = jobId, status = "PENDING")
    }

    override suspend fun getJobStatus(jobId: String): JobStatusResponse {
        delay(250)
        val entry = jobs[jobId] ?: return JobStatusResponse(
            status = JobStatus.FAILED,
            progress = 0,
            errorMessage = "Job introuvable"
        )
        val elapsed = System.currentTimeMillis() - entry.createdAt
        val progress = ((elapsed / 8000.0) * 100).toInt().coerceIn(0, 100)
        val status = if (progress >= 100) JobStatus.DONE else JobStatus.RUNNING
        return JobStatusResponse(
            status = status,
            progress = progress,
            stage = if (status == JobStatus.DONE) "DONE" else "RUNNING"
        )
    }

    override suspend fun getJobResult(jobId: String): JobResultResponse {
        delay(200)
        val entry = jobs[jobId]
        val outputType = entry?.outputType ?: "both"
        val musicXmlUrl = if (outputType == "score" || outputType == "both") {
            "fake://$jobId/musicxml"
        } else {
            null
        }
        val tabUrl = if (outputType == "tab" || outputType == "both") {
            "fake://$jobId/tab"
        } else {
            null
        }
        return JobResultResponse(
            musicXmlUrl = musicXmlUrl,
            tabTxtUrl = tabUrl,
            metadata = JobMetadata(durationSeconds = entry?.durationSeconds?.toDouble())
        )
    }

    override suspend fun deleteJob(jobId: String) {
        jobs.remove(jobId)
    }
}
