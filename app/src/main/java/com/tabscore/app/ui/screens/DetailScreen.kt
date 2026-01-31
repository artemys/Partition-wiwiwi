package com.tabscore.app.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.tabscore.app.R
import com.tabscore.app.domain.model.OutputType
import com.tabscore.app.domain.model.TranscriptionStatus
import com.tabscore.app.ui.util.ShareHelper
import com.tabscore.app.ui.viewmodel.DetailViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DetailScreen(
    viewModel: DetailViewModel,
    transcriptionId: UUID,
    onBack: () -> Unit
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val transcription by viewModel.transcription.collectAsState()
    val previewText = remember { mutableStateOf("") }

    LaunchedEffect(transcriptionId) {
        viewModel.load(transcriptionId)
    }

    LaunchedEffect(transcription?.resultMusicXmlPath, transcription?.resultTabPath) {
        val path = transcription?.resultMusicXmlPath ?: transcription?.resultTabPath
        previewText.value = if (path != null) {
            withContext(Dispatchers.IO) {
                File(path).takeIf { it.exists() }?.readText() ?: ""
            }
        } else {
            ""
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(id = R.string.detail_title)) },
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
            Text(text = transcription?.title ?: "")
            if (transcription?.status == TranscriptionStatus.PENDING ||
                transcription?.status == TranscriptionStatus.RUNNING
            ) {
                LinearProgressIndicator(
                    progress = { (transcription?.progress ?: 0) / 100f },
                    modifier = Modifier.fillMaxWidth()
                )
                Text(text = transcription?.stage ?: "Traitement en cours")
            }
            Text(
                text = when (transcription?.outputType) {
                    OutputType.SCORE -> stringResource(id = R.string.preview_score)
                    OutputType.TAB -> stringResource(id = R.string.preview_tab)
                    OutputType.BOTH -> stringResource(id = R.string.preview_both)
                    else -> ""
                }
            )
            Text(
                text = previewText.value,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(200.dp)
            )
            Spacer(modifier = Modifier.height(8.dp))
            Button(onClick = {
                scope.launch {
                    val paths = viewModel.download(transcriptionId)
                    ShareHelper.shareFiles(context, paths)
                }
            }) {
                Text(text = stringResource(id = R.string.download))
            }
            Button(onClick = {
                viewModel.delete(transcriptionId, onBack)
            }) {
                Text(text = stringResource(id = R.string.delete))
            }
        }
    }
}
