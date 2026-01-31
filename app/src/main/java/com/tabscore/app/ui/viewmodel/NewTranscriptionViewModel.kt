package com.tabscore.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tabscore.app.data.settings.SettingsRepository
import com.tabscore.app.domain.model.GuitarMode
import com.tabscore.app.domain.model.GuitarTuning
import com.tabscore.app.domain.model.OutputType
import com.tabscore.app.domain.model.Quality
import com.tabscore.app.domain.model.SourceType
import com.tabscore.app.domain.model.TranscriptionStatus
import com.tabscore.app.domain.repository.TranscriptionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

data class NewTranscriptionUiState(
    val sourceType: SourceType = SourceType.AUDIO_FILE,
    val audioUri: String? = null,
    val youtubeUrl: String = "",
    val outputType: OutputType = OutputType.BOTH,
    val instrument: String = "Guitare",
    val tuning: GuitarTuning = GuitarTuning.STANDARD,
    val capo: Int = 0,
    val mode: GuitarMode = GuitarMode.BEST_EFFORT,
    val quality: Quality = Quality.FAST,
    val status: TranscriptionStatus = TranscriptionStatus.PENDING,
    val progress: Int = 0,
    val stage: String? = null,
    val errorMessage: String? = null,
    val createdId: UUID? = null,
    val hasStarted: Boolean = false
)

@HiltViewModel
class NewTranscriptionViewModel @Inject constructor(
    private val repository: TranscriptionRepository,
    private val settingsRepository: SettingsRepository
) : ViewModel() {
    private val _state = MutableStateFlow(NewTranscriptionUiState())
    val state: StateFlow<NewTranscriptionUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            val quality = settingsRepository.quality.first()
            _state.value = _state.value.copy(quality = quality)
        }
    }

    fun setSourceType(type: SourceType) {
        _state.value = _state.value.copy(sourceType = type)
    }

    fun setAudioUri(uri: String?) {
        _state.value = _state.value.copy(audioUri = uri)
    }

    fun setYoutubeUrl(url: String) {
        _state.value = _state.value.copy(youtubeUrl = url)
    }

    fun setOutputType(type: OutputType) {
        _state.value = _state.value.copy(outputType = type)
    }

    fun setInstrument(value: String) {
        _state.value = _state.value.copy(instrument = value)
    }

    fun setTuning(tuning: GuitarTuning) {
        _state.value = _state.value.copy(tuning = tuning)
    }

    fun setCapo(capo: Int) {
        _state.value = _state.value.copy(capo = capo.coerceIn(0, 12))
    }

    fun setMode(mode: GuitarMode) {
        _state.value = _state.value.copy(mode = mode)
    }

    fun setQuality(quality: Quality) {
        _state.value = _state.value.copy(quality = quality)
    }

    fun startTranscription(title: String, onCreated: (UUID) -> Unit) {
        val current = _state.value
        viewModelScope.launch {
            try {
                _state.value = current.copy(
                    status = TranscriptionStatus.PENDING,
                    progress = 0,
                    stage = "PENDING",
                    hasStarted = true,
                    errorMessage = null
                )
                val id = if (current.sourceType == SourceType.AUDIO_FILE) {
                    val uri = current.audioUri ?: error("Fichier audio manquant")
                    repository.createFromAudio(
                        audioUri = uri,
                        title = title,
                        outputType = current.outputType,
                        instrument = current.instrument,
                        tuning = current.tuning,
                        capo = current.capo,
                        mode = current.mode,
                        quality = current.quality
                    )
                } else {
                    if (!current.youtubeUrl.startsWith("http")) {
                        error("URL YouTube invalide")
                    }
                    repository.createFromYoutube(
                        youtubeUrl = current.youtubeUrl,
                        title = title,
                        outputType = current.outputType,
                        instrument = current.instrument,
                        tuning = current.tuning,
                        capo = current.capo,
                        mode = current.mode,
                        quality = current.quality
                    )
                }
                _state.value = _state.value.copy(
                    status = TranscriptionStatus.RUNNING,
                    progress = 10,
                    createdId = id,
                    errorMessage = null
                )
                observeTranscription(id)
                onCreated(id)
            } catch (t: Throwable) {
                _state.value = _state.value.copy(
                    status = TranscriptionStatus.FAILED,
                    errorMessage = t.message,
                    hasStarted = true
                )
            }
        }
    }

    private fun observeTranscription(id: UUID) {
        viewModelScope.launch {
            repository.observeById(id).collectLatest { transcription ->
                transcription ?: return@collectLatest
                _state.value = _state.value.copy(
                    status = transcription.status,
                    progress = transcription.progress,
                    stage = transcription.stage,
                    errorMessage = transcription.errorMessage
                )
            }
        }
    }
}
