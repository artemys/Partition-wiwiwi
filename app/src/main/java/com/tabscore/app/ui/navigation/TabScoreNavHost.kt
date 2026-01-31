package com.tabscore.app.ui.navigation

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.tabscore.app.ui.screens.DetailScreen
import com.tabscore.app.ui.screens.LibraryScreen
import com.tabscore.app.ui.screens.NewTranscriptionScreen
import com.tabscore.app.ui.screens.SettingsScreen
import com.tabscore.app.ui.viewmodel.DetailViewModel
import com.tabscore.app.ui.viewmodel.LibraryViewModel
import com.tabscore.app.ui.viewmodel.NewTranscriptionViewModel
import com.tabscore.app.ui.viewmodel.SettingsViewModel
import java.util.UUID

@Composable
fun TabScoreNavHost(
    modifier: Modifier = Modifier,
    navController: NavHostController = rememberNavController()
) {
    NavHost(
        navController = navController,
        startDestination = Routes.Library,
        modifier = modifier
    ) {
        composable(Routes.Library) {
            val viewModel = hiltViewModel<LibraryViewModel>()
            LibraryScreen(
                viewModel = viewModel,
                onNew = { navController.navigate(Routes.NewTranscription) },
                onOpen = { id -> navController.navigate("${Routes.Detail}/$id") },
                onSettings = { navController.navigate(Routes.Settings) }
            )
        }
        composable(Routes.NewTranscription) {
            val viewModel = hiltViewModel<NewTranscriptionViewModel>()
            NewTranscriptionScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() },
                onOpenDetail = { id -> navController.navigate("${Routes.Detail}/$id") }
            )
        }
        composable("${Routes.Detail}/{id}") { backStackEntry ->
            val id = UUID.fromString(backStackEntry.arguments?.getString("id"))
            val viewModel = hiltViewModel<DetailViewModel>()
            DetailScreen(
                viewModel = viewModel,
                transcriptionId = id,
                onBack = { navController.popBackStack() }
            )
        }
        composable(Routes.Settings) {
            val viewModel = hiltViewModel<SettingsViewModel>()
            SettingsScreen(
                viewModel = viewModel,
                onBack = { navController.popBackStack() }
            )
        }
    }
}
