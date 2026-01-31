package com.tabscore.app.data.remote

import com.tabscore.app.data.remote.model.CreateJobResponse
import com.tabscore.app.data.remote.model.JobResultResponse
import com.tabscore.app.data.remote.model.JobStatusResponse

interface TabScoreRemoteApi {
    suspend fun createJobFromAudio(
        audioBytes: ByteArray,
        filename: String,
        mimeType: String,
        targetInstrument: String,
        target: String,
        outputType: String,
        tuning: String,
        capo: Int,
        mode: String,
        quality: String
    ): CreateJobResponse

    suspend fun createJobFromYoutube(
        youtubeUrl: String,
        targetInstrument: String,
        target: String,
        outputType: String,
        tuning: String,
        capo: Int,
        mode: String,
        quality: String
    ): CreateJobResponse

    suspend fun getJobStatus(jobId: String): JobStatusResponse
    suspend fun getJobResult(jobId: String): JobResultResponse
    suspend fun deleteJob(jobId: String)
}
