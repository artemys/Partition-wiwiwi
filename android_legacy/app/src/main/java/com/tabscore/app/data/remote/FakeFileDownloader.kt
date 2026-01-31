package com.tabscore.app.data.remote

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class FakeFileDownloader(
    private val context: Context
) : FileDownloader {
    override suspend fun download(url: String): ByteArray = withContext(Dispatchers.IO) {
        if (url.contains("musicxml")) {
            context.assets.open("sample_musicxml.xml").use { it.readBytes() }
        } else {
            context.assets.open("sample_tab.txt").use { it.readBytes() }
        }
    }
}
