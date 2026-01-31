package com.tabscore.app.data.remote.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class CreateJobResponse(
    val jobId: String,
    val status: String
)

@Serializable
data class JobStatusResponse(
    val status: JobStatus,
    val progress: Int,
    val stage: String? = null,
    val errorMessage: String? = null,
    val confidence: Float? = null,
    val createdAt: String? = null,
    val warnings: List<String> = emptyList()
)

@Serializable
data class JobResultResponse(
    val tabTxtUrl: String? = null,
    val tabJsonUrl: String? = null,
    val musicXmlUrl: String? = null,
    val midiUrl: String? = null,
    val metadata: JobMetadata? = null,
    val warnings: List<String> = emptyList()
)

@Serializable
data class JobMetadata(
    val durationSeconds: Double? = null,
    val tuning: String? = null,
    val capo: Int? = null,
    val quality: String? = null,
    val tempoBpm: Double? = null
)

@Serializable
enum class JobStatus {
    @SerialName("PENDING") PENDING,
    @SerialName("RUNNING") RUNNING,
    @SerialName("DONE") DONE,
    @SerialName("FAILED") FAILED
}

@Serializable
data class YoutubeRequest(
    val youtubeUrl: String
)
