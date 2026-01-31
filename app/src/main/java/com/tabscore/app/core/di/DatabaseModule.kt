package com.tabscore.app.core.di

import android.content.Context
import androidx.room.Room
import com.tabscore.app.data.local.TabScoreDatabase
import com.tabscore.app.data.local.TranscriptionDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object DatabaseModule {
    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): TabScoreDatabase =
        Room.databaseBuilder(context, TabScoreDatabase::class.java, "tabscore.db")
            .fallbackToDestructiveMigration()
            .build()

    @Provides
    fun provideTranscriptionDao(db: TabScoreDatabase): TranscriptionDao = db.transcriptionDao()
}
