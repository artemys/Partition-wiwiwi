package com.tabscore.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tabscore.app.domain.model.Transcription
import com.tabscore.app.domain.repository.TranscriptionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

@HiltViewModel
class DetailViewModel @Inject constructor(
    private val repository: TranscriptionRepository
) : ViewModel() {
    private val _transcription = MutableStateFlow<Transcription?>(null)
    val transcription: StateFlow<Transcription?> = _transcription.asStateFlow()

    fun load(id: UUID) {
        viewModelScope.launch {
            repository.observeById(id).collectLatest {
                _transcription.value = it
            }
        }
    }

    fun delete(id: UUID, onDone: () -> Unit) {
        viewModelScope.launch {
            repository.delete(id)
            onDone()
        }
    }

    suspend fun download(id: UUID): List<String> = repository.downloadExports(id)

    fun retry(id: UUID) {
        viewModelScope.launch { repository.retry(id) }
    }
}
