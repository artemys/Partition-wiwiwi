package com.tabscore.app.data.settings

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.tabscore.app.domain.model.ExportFormat
import com.tabscore.app.domain.model.Quality
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "settings")

interface SettingsRepository {
    val exportFormat: Flow<ExportFormat>
    val quality: Flow<Quality>
    suspend fun setExportFormat(format: ExportFormat)
    suspend fun setQuality(quality: Quality)
}

class DataStoreSettingsRepository(
    private val context: Context
) : SettingsRepository {
    private val exportKey = stringPreferencesKey("export_format")
    private val qualityKey = stringPreferencesKey("quality")

    override val exportFormat: Flow<ExportFormat> = context.dataStore.data.map { prefs ->
        prefs[exportKey]?.let { ExportFormat.valueOf(it) } ?: ExportFormat.MUSICXML
    }

    override val quality: Flow<Quality> = context.dataStore.data.map { prefs ->
        prefs[qualityKey]?.let { Quality.valueOf(it) } ?: Quality.FAST
    }

    override suspend fun setExportFormat(format: ExportFormat) {
        context.dataStore.edit { prefs -> prefs[exportKey] = format.name }
    }

    override suspend fun setQuality(quality: Quality) {
        context.dataStore.edit { prefs -> prefs[qualityKey] = quality.name }
    }
}
