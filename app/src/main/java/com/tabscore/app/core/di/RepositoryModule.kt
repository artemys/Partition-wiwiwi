package com.tabscore.app.core.di

import android.content.Context
import com.tabscore.app.data.local.TranscriptionDao
import com.tabscore.app.data.remote.FileDownloader
import com.tabscore.app.data.remote.TabScoreRemoteApi
import com.tabscore.app.data.repository.TranscriptionRepositoryImpl
import com.tabscore.app.data.settings.DataStoreSettingsRepository
import com.tabscore.app.data.settings.SettingsRepository
import com.tabscore.app.data.storage.AudioReader
import com.tabscore.app.data.storage.ContentResolverAudioReader
import com.tabscore.app.data.storage.ExportStorage
import com.tabscore.app.data.storage.FileExportStorage
import com.tabscore.app.domain.repository.TranscriptionRepository
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object RepositoryModule {
    @Provides
    @Singleton
    fun provideAudioReader(@ApplicationContext context: Context): AudioReader =
        ContentResolverAudioReader(context)

    @Provides
    @Singleton
    fun provideExportStorage(@ApplicationContext context: Context): ExportStorage =
        FileExportStorage(context)

    @Provides
    @Singleton
    fun provideTranscriptionRepository(
        dao: TranscriptionDao,
        remoteApi: TabScoreRemoteApi,
        fileDownloader: FileDownloader,
        exportStorage: ExportStorage,
        audioReader: AudioReader,
        settingsRepository: SettingsRepository,
        @ApplicationScope appScope: CoroutineScope,
        ioDispatcher: CoroutineDispatcher
    ): TranscriptionRepository = TranscriptionRepositoryImpl(
        dao = dao,
        remoteApi = remoteApi,
        fileDownloader = fileDownloader,
        exportStorage = exportStorage,
        audioReader = audioReader,
        settingsRepository = settingsRepository,
        appScope = appScope,
        ioDispatcher = ioDispatcher
    )

    @Provides
    @Singleton
    fun provideSettingsRepository(@ApplicationContext context: Context): SettingsRepository =
        DataStoreSettingsRepository(context)
}
