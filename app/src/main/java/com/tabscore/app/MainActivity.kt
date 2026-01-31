package com.tabscore.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import com.tabscore.app.ui.navigation.TabScoreNavHost
import com.tabscore.app.ui.theme.TabScoreTheme
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            TabScoreTheme {
                Surface {
                    TabScoreNavHost()
                }
            }
        }
    }
}
