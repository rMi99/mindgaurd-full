"use client"

import React, { useEffect, useRef, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Mic, MicOff } from "lucide-react"

export interface AudioEmotionPoint {
  timestamp: number
  emotions: Record<string, number>
  transcript?: string | null
}

interface VoiceMonitorProps {
  isActive: boolean
  onData: (point: AudioEmotionPoint) => void
}

export default function VoiceMonitor({ isActive, onData }: VoiceMonitorProps) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const [permissionGranted, setPermissionGranted] = useState(false)
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!isActive) {
      stopRecording()
      return
    }
    startRecording()
    return () => stopRecording()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive])

  async function requestMic(): Promise<MediaStream | null> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      setPermissionGranted(true)
      return stream
    } catch (e) {
      setPermissionGranted(false)
      setError("Microphone permission denied")
      return null
    }
  }

  async function startRecording() {
    if (recording) return
    const stream = await requestMic()
    if (!stream) return

    const mimeType =
      typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/webm;codecs=opus"

    const recorder = new MediaRecorder(stream, { mimeType })
    mediaRecorderRef.current = recorder
    const chunks: Blob[] = []

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        chunks.push(e.data)
      }
    }

    recorder.onstop = async () => {
      if (chunks.length === 0) return
      const blob = new Blob(chunks, { type: mimeType })
      const reader = new FileReader()
      reader.onloadend = async () => {
        const base64data = reader.result as string
        try {
          const res = await fetch("/api/audio-analysis", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio: base64data, contentType: mimeType }),
          })
          if (res.ok) {
            const result = await res.json()
            onData({
              timestamp: Date.now(),
              emotions: result.emotions || {},
              transcript: result.transcript || null,
            })
          }
        } catch (err) {
          // ignore chunk errors to keep UX smooth
        }
      }
      reader.readAsDataURL(blob)
    }

    recorder.start(3000) // timeslice: capture chunks every 3s
    setRecording(true)
  }

  function stopRecording() {
    const recorder = mediaRecorderRef.current
    if (recorder && recorder.state !== "inactive") {
      recorder.stop()
    }
    mediaRecorderRef.current = null
    setRecording(false)
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Voice Monitoring</span>
          <div className="flex items-center gap-2">
            {recording ? (
              <Badge variant="default" className="animate-pulse">Active</Badge>
            ) : (
              <Badge variant="secondary">Idle</Badge>
            )}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {error && (
          <Alert>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        <div className="flex gap-2">
          {!recording ? (
            <Button onClick={startRecording}>
              <Mic className="h-4 w-4 mr-2" /> Start
            </Button>
          ) : (
            <Button variant="outline" onClick={stopRecording}>
              <MicOff className="h-4 w-4 mr-2" /> Stop
            </Button>
          )}
        </div>
        <p className="text-xs text-gray-500">Audio is processed in real-time; nothing is stored.</p>
      </CardContent>
    </Card>
  )
}



