package com.tabscore.app.data.local

import androidx.room.TypeConverter
import com.tabscore.app.domain.model.OutputType
import com.tabscore.app.domain.model.SourceType
import com.tabscore.app.domain.model.TranscriptionStatus
import com.tabscore.app.domain.model.GuitarMode
import com.tabscore.app.domain.model.GuitarTuning
import java.util.UUID

class Converters {
    @TypeConverter
    fun uuidToString(value: UUID?): String? = value?.toString()

    @TypeConverter
    fun stringToUuid(value: String?): UUID? = value?.let(UUID::fromString)

    @TypeConverter
    fun sourceTypeToString(value: SourceType?): String? = value?.name

    @TypeConverter
    fun stringToSourceType(value: String?): SourceType? = value?.let(SourceType::valueOf)

    @TypeConverter
    fun outputTypeToString(value: OutputType?): String? = value?.name

    @TypeConverter
    fun stringToOutputType(value: String?): OutputType? = value?.let(OutputType::valueOf)

    @TypeConverter
    fun statusToString(value: TranscriptionStatus?): String? = value?.name

    @TypeConverter
    fun stringToStatus(value: String?): TranscriptionStatus? = value?.let(TranscriptionStatus::valueOf)

    @TypeConverter
    fun tuningToString(value: GuitarTuning?): String? = value?.name

    @TypeConverter
    fun stringToTuning(value: String?): GuitarTuning? = value?.let(GuitarTuning::valueOf)

    @TypeConverter
    fun modeToString(value: GuitarMode?): String? = value?.name

    @TypeConverter
    fun stringToMode(value: String?): GuitarMode? = value?.let(GuitarMode::valueOf)
}
