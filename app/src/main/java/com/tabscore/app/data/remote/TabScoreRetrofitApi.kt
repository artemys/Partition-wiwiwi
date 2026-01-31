package com.tabscore.app.data.remote

import com.tabscore.app.data.remote.model.CreateJobResponse
import com.tabscore.app.data.remote.model.JobResultResponse
import com.tabscore.app.data.remote.model.JobStatusResponse
import com.tabscore.app.data.remote.model.YoutubeRequest
import okhttp3.MultipartBody
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path
import retrofit2.http.Query
import retrofit2.http.DELETE

interface TabScoreRetrofitApi {
    @Multipart
    @POST("jobs")
    suspend fun createJobFromAudio(
        @Part audio: MultipartBody.Part,
        @Query("targetInstrument") targetInstrument: String,
        @Query("target") target: String,
        @Query("outputType") outputType: String,
        @Query("tuning") tuning: String,
        @Query("capo") capo: Int,
        @Query("mode") mode: String,
        @Query("quality") quality: String,
        @Query("startSeconds") startSeconds: Int?,
        @Query("endSeconds") endSeconds: Int?
    ): CreateJobResponse

    @POST("jobs")
    suspend fun createJobFromYoutube(
        @Body body: YoutubeRequest,
        @Query("targetInstrument") targetInstrument: String,
        @Query("target") target: String,
        @Query("outputType") outputType: String,
        @Query("tuning") tuning: String,
        @Query("capo") capo: Int,
        @Query("mode") mode: String,
        @Query("quality") quality: String,
        @Query("startSeconds") startSeconds: Int?,
        @Query("endSeconds") endSeconds: Int?
    ): CreateJobResponse

    @GET("jobs/{jobId}")
    suspend fun getJobStatus(@Path("jobId") jobId: String): JobStatusResponse

    @GET("jobs/{jobId}/result")
    suspend fun getJobResult(@Path("jobId") jobId: String): JobResultResponse

    @DELETE("jobs/{jobId}")
    suspend fun deleteJob(@Path("jobId") jobId: String)
}
