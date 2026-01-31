package com.tabscore.app.core.di

import android.content.Context
import com.tabscore.app.BuildConfig
import com.tabscore.app.data.remote.FileDownloader
import com.tabscore.app.data.remote.OkHttpFileDownloader
import com.tabscore.app.data.remote.RealTabScoreApi
import com.tabscore.app.data.remote.TabScoreRemoteApi
import com.tabscore.app.data.remote.TabScoreRetrofitApi
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object NetworkModule {

    @Provides
    @Singleton
    fun provideJson(): Json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
    }

    @Provides
    @Singleton
    fun provideOkHttp(): OkHttpClient {
        val logging = HttpLoggingInterceptor().apply {
            level = if (BuildConfig.DEBUG) HttpLoggingInterceptor.Level.BODY
            else HttpLoggingInterceptor.Level.NONE
        }
        return OkHttpClient.Builder()
            .addInterceptor(logging)
            .build()
    }

    @Provides
    @Singleton
    fun provideRetrofit(okHttpClient: OkHttpClient, json: Json): Retrofit {
        val contentType = "application/json".toMediaType()
        return Retrofit.Builder()
            .baseUrl(BuildConfig.BASE_URL)
            .client(okHttpClient)
            .addConverterFactory(json.asConverterFactory(contentType))
            .build()
    }

    @Provides
    @Singleton
    fun provideRetrofitApi(retrofit: Retrofit): TabScoreRetrofitApi =
        retrofit.create(TabScoreRetrofitApi::class.java)

    @Provides
    @Singleton
    fun provideRemoteApi(retrofitApi: TabScoreRetrofitApi): TabScoreRemoteApi =
        RealTabScoreApi(retrofitApi)

    @Provides
    @Singleton
    fun provideFileDownloader(
        okHttpClient: OkHttpClient,
        @ApplicationContext context: Context
    ): FileDownloader = OkHttpFileDownloader(okHttpClient)
}
