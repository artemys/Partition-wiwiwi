package com.tabscore.app.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import com.tabscore.app.R
import com.tabscore.app.domain.model.ExportFormat
import com.tabscore.app.domain.model.Quality
import com.tabscore.app.ui.util.displayName
import com.tabscore.app.ui.viewmodel.SettingsViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    viewModel: SettingsViewModel,
    onBack: () -> Unit
) {
    val state by viewModel.state.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(id = R.string.settings)) },
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
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(text = stringResource(id = R.string.export_format))
            ExportFormat.values().forEach { format ->
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    RadioButton(
                        selected = state.exportFormat == format,
                        onClick = { viewModel.setExportFormat(format) }
                    )
                    Text(text = format.displayName())
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
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
        }
    }
}
