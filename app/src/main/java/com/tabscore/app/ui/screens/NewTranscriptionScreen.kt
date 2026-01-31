package com.tabscore.app.ui.screens

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import com.tabscore.app.R
import com.tabscore.app.domain.model.OutputType
import com.tabscore.app.domain.model.Quality
import com.tabscore.app.domain.model.SourceType
import com.tabscore.app.domain.model.TranscriptionStatus
import com.tabscore.app.domain.model.GuitarMode
import com.tabscore.app.domain.model.GuitarTuning
import com.tabscore.app.ui.viewmodel.NewTranscriptionViewModel
import kotlinx.coroutines.launch
import java.util.UUID

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun NewTranscriptionScreen(
    viewModel: NewTranscriptionViewModel,
    onBack: () -> Unit,
    onOpenDetail: (UUID) -> Unit
) {
    val state by viewModel.state.collectAsState()
    val scope = rememberCoroutineScope()
    val titleInput = remember { mutableStateOf("") }

    val filePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        viewModel.setAudioUri(uri?.toString())
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(id = R.string.new_transcription)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = stringResource(id = R.string.cancel)
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors()
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(text = stringResource(id = R.string.source_audio))
            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(
                    selected = state.sourceType == SourceType.AUDIO_FILE,
                    onClick = { viewModel.setSourceType(SourceType.AUDIO_FILE) }
                )
                Text(text = stringResource(id = R.string.source_audio))
                Spacer(modifier = Modifier.width(16.dp))
                RadioButton(
                    selected = state.sourceType == SourceType.YOUTUBE_URL,
                    onClick = { viewModel.setSourceType(SourceType.YOUTUBE_URL) }
                )
                Text(text = stringResource(id = R.string.source_youtube))
            }

            if (state.sourceType == SourceType.AUDIO_FILE) {
                Button(onClick = { filePicker.launch(arrayOf("audio/*")) }) {
                    Text(text = stringResource(id = R.string.choose_file))
                }
                Text(text = state.audioUri ?: stringResource(id = R.string.unknown))
            } else {
                OutlinedTextField(
                    value = state.youtubeUrl,
                    onValueChange = viewModel::setYoutubeUrl,
                    label = { Text(stringResource(id = R.string.youtube_url_hint)) },
                    modifier = Modifier.fillMaxWidth()
                )
            }

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                OutlinedTextField(
                    value = state.startSecondsInput,
                    onValueChange = viewModel::setStartSeconds,
                    label = { Text(text = "Debut (s)") },
                    modifier = Modifier.weight(1f)
                )
                OutlinedTextField(
                    value = state.endSecondsInput,
                    onValueChange = viewModel::setEndSeconds,
                    label = { Text(text = "Fin (s)") },
                    modifier = Modifier.weight(1f)
                )
            }

            HorizontalDivider()

            OutlinedTextField(
                value = titleInput.value,
                onValueChange = { titleInput.value = it },
                label = { Text(stringResource(id = R.string.title_label)) },
                modifier = Modifier.fillMaxWidth()
            )

            Text(text = stringResource(id = R.string.instrument_target))
            Text(text = state.instrument)

            Text(text = stringResource(id = R.string.output_label))
            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(
                    selected = state.outputType == OutputType.SCORE,
                    onClick = { viewModel.setOutputType(OutputType.SCORE) }
                )
                Text(text = stringResource(id = R.string.output_score))
                Spacer(modifier = Modifier.width(16.dp))
                RadioButton(
                    selected = state.outputType == OutputType.TAB,
                    onClick = { viewModel.setOutputType(OutputType.TAB) }
                )
                Text(text = stringResource(id = R.string.output_tab))
                Spacer(modifier = Modifier.width(16.dp))
                RadioButton(
                    selected = state.outputType == OutputType.BOTH,
                    onClick = { viewModel.setOutputType(OutputType.BOTH) }
                )
                Text(text = stringResource(id = R.string.output_both))
            }

            Text(text = stringResource(id = R.string.tuning_label))
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                listOf(
                    GuitarTuning.STANDARD,
                    GuitarTuning.DROP_D,
                    GuitarTuning.OPEN_G
                ).forEach { tuning ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(
                            selected = state.tuning == tuning,
                            onClick = { viewModel.setTuning(tuning) }
                        )
                        Text(text = tuning.displayName)
                    }
                }
            }

            Text(text = stringResource(id = R.string.capo_label))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Button(onClick = { viewModel.setCapo(state.capo - 1) }) {
                    Text(text = "-")
                }
                Spacer(modifier = Modifier.width(12.dp))
                Text(text = state.capo.toString())
                Spacer(modifier = Modifier.width(12.dp))
                Button(onClick = { viewModel.setCapo(state.capo + 1) }) {
                    Text(text = "+")
                }
            }

            Text(text = stringResource(id = R.string.mode_label))
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                listOf(GuitarMode.BEST_EFFORT, GuitarMode.ISOLATED_TRACK).forEach { mode ->
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(
                            selected = state.mode == mode,
                            onClick = { viewModel.setMode(mode) }
                        )
                        Text(text = mode.displayName)
                    }
                }
            }

            Text(text = stringResource(id = R.string.quality))
            Row(verticalAlignment = Alignment.CenterVertically) {
                RadioButton(
                    selected = state.quality == Quality.FAST,
                    onClick = { viewModel.setQuality(Quality.FAST) }
                )
                Text(text = stringResource(id = R.string.quality_fast))
                Spacer(modifier = Modifier.width(16.dp))
                RadioButton(
                    selected = state.quality == Quality.ACCURATE,
                    onClick = { viewModel.setQuality(Quality.ACCURATE) }
                )
                Text(text = stringResource(id = R.string.quality_accurate))
            }

            Button(
                onClick = {
                    scope.launch {
                        val title = titleInput.value.ifBlank { "Transcription" }
                        viewModel.startTranscription(title, onOpenDetail)
                    }
                },
                enabled = state.status != TranscriptionStatus.RUNNING
            ) {
                Text(text = stringResource(id = R.string.transcribe))
            }

            if (state.hasStarted &&
                (state.status == TranscriptionStatus.RUNNING || state.status == TranscriptionStatus.PENDING)
            ) {
                LinearProgressIndicator(
                    progress = { state.progress / 100f },
                    modifier = Modifier.fillMaxWidth()
                )
                state.stage?.let { stage ->
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(text = stage)
                }
            }

            if (!state.hasStarted) {
                Text(text = stringResource(id = R.string.status_idle))
            } else {
                when (state.status) {
                    TranscriptionStatus.PENDING -> Text(text = stringResource(id = R.string.status_uploading))
                    TranscriptionStatus.RUNNING -> Text(text = stringResource(id = R.string.status_processing))
                    TranscriptionStatus.DONE -> Text(text = stringResource(id = R.string.status_done))
                    TranscriptionStatus.FAILED -> Text(text = state.errorMessage ?: stringResource(id = R.string.error_generic))
                }
            }
        }
    }
}
