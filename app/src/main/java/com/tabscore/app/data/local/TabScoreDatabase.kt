package com.tabscore.app.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverters

@Database(
    entities = [TranscriptionEntity::class],
    version = 4,
    exportSchema = false
)
@TypeConverters(Converters::class)
abstract class TabScoreDatabase : RoomDatabase() {
    abstract fun transcriptionDao(): TranscriptionDao
}
