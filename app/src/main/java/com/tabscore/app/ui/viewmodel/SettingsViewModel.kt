package com.tabscore.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tabscore.app.data.settings.SettingsRepository
import com.tabscore.app.domain.model.ExportFormat
import com.tabscore.app.domain.model.Quality
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.launch
import javax.inject.Inject

data class SettingsUiState(
    val exportFormat: ExportFormat = ExportFormat.MUSICXML,
    val quality: Quality = Quality.ACCURATE
)

@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val settingsRepository: SettingsRepository
) : ViewModel() {
    private val _state = MutableStateFlow(SettingsUiState())
    val state: StateFlow<SettingsUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            combine(
                settingsRepository.exportFormat,
                settingsRepository.quality
            ) { export, quality -> SettingsUiState(export, quality) }
                .collect { _state.value = it }
        }
    }

    fun setExportFormat(format: ExportFormat) {
        viewModelScope.launch { settingsRepository.setExportFormat(format) }
    }

    fun setQuality(quality: Quality) {
        viewModelScope.launch { settingsRepository.setQuality(quality) }
    }
}
