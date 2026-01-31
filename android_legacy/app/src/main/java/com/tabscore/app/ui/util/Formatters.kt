package com.tabscore.app.ui.util

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object Formatters {
    private val dateFormat = SimpleDateFormat("dd/MM/yyyy HH:mm", Locale.FRANCE)

    fun formatDate(timestamp: Long): String = dateFormat.format(Date(timestamp))

    fun formatDuration(seconds: Int?): String {
        if (seconds == null) return "—"
        val min = seconds / 60
        val sec = seconds % 60
        return "%d:%02d".format(min, sec)
    }
}
