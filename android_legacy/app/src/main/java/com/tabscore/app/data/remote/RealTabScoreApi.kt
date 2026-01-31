package com.tabscore.app.data.remote

import com.tabscore.app.data.remote.model.CreateJobResponse
import com.tabscore.app.data.remote.model.JobResultResponse
import com.tabscore.app.data.remote.model.JobStatusResponse
import com.tabscore.app.data.remote.model.YoutubeRequest
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

class RealTabScoreApi(
    private val retrofitApi: TabScoreRetrofitApi
) : TabScoreRemoteApi {
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
        val body = audioBytes.toRequestBody(mimeType.toMediaType())
        val part = MultipartBody.Part.createFormData("audio", filename, body)
        return retrofitApi.createJobFromAudio(
            audio = part,
            targetInstrument = targetInstrument,
            target = target,
            outputType = outputType,
            tuning = tuning,
            capo = capo,
            mode = mode,
            quality = quality,
            startSeconds = startSeconds,
            endSeconds = endSeconds
        )
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
        return retrofitApi.createJobFromYoutube(
            body = YoutubeRequest(youtubeUrl),
            targetInstrument = targetInstrument,
            target = target,
            outputType = outputType,
            tuning = tuning,
            capo = capo,
            mode = mode,
            quality = quality,
            startSeconds = startSeconds,
            endSeconds = endSeconds
        )
    }

    override suspend fun getJobStatus(jobId: String): JobStatusResponse =
        retrofitApi.getJobStatus(jobId)

    override suspend fun getJobResult(jobId: String): JobResultResponse =
        retrofitApi.getJobResult(jobId)

    override suspend fun deleteJob(jobId: String) {
        retrofitApi.deleteJob(jobId)
    }
}
