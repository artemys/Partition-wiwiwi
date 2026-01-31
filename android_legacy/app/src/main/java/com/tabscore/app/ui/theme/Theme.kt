package com.tabscore.app.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val LightColors = lightColorScheme(
    primary = Primary,
    secondary = Secondary,
    surface = Surface
)

private val DarkColors = darkColorScheme(
    primary = Primary,
    secondary = Secondary,
    surface = Color(0xFF121212)
)

@Composable
fun TabScoreTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = LightColors,
        typography = TabScoreTypography,
        content = content
    )
}
