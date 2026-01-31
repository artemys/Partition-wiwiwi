package com.tabscore.app.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.tabscore.app.domain.model.Transcription
import com.tabscore.app.domain.repository.TranscriptionRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.util.UUID
import javax.inject.Inject

@HiltViewModel
class LibraryViewModel @Inject constructor(
    private val repository: TranscriptionRepository
) : ViewModel() {
    val transcriptions: StateFlow<List<Transcription>> =
        repository.observeAll()
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    fun delete(id: UUID) {
        viewModelScope.launch { repository.delete(id) }
    }

    suspend fun download(id: UUID): List<String> {
        return repository.downloadExports(id)
    }
}
