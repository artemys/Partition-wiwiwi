package com.tabscore.app.data.local

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow
import java.util.UUID

@Dao
interface TranscriptionDao {
    @Query("SELECT * FROM transcriptions ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<TranscriptionEntity>>

    @Query("SELECT * FROM transcriptions WHERE id = :id")
    fun observeById(id: UUID): Flow<TranscriptionEntity?>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(entity: TranscriptionEntity)

    @Update
    suspend fun update(entity: TranscriptionEntity)

    @Delete
    suspend fun delete(entity: TranscriptionEntity)

    @Query("SELECT * FROM transcriptions WHERE id = :id")
    suspend fun getById(id: UUID): TranscriptionEntity?
}
