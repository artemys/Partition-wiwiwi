package com.tabscore.app.ui.util

import com.tabscore.app.domain.model.ExportFormat
import com.tabscore.app.domain.model.Quality

fun ExportFormat.displayName(): String = when (this) {
    ExportFormat.PDF -> "PDF"
    ExportFormat.MUSICXML -> "MusicXML"
    ExportFormat.TXT -> "TXT"
}

fun Quality.displayName(): String = when (this) {
    Quality.FAST -> "Rapide"
    Quality.ACCURATE -> "Précise"
}
