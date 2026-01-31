package com.tabscore.app.data.remote

interface FileDownloader {
    suspend fun download(url: String): ByteArray
}
