"use client"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Play, Pause, RotateCcw, Volume2, Download } from "lucide-react"

interface AudioTrack {
  title: string
  duration: string
  type: string
}

interface AudioPlayerProps {
  tracks: AudioTrack[]
  riskLevel: string
}

export default function AudioPlayer({ tracks, riskLevel }: AudioPlayerProps) {
  const [currentTrack, setCurrentTrack] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentTime, setCurrentTime] = useState("0:00")
  const audioRef = useRef<HTMLAudioElement>(null)

  // Mock audio URLs - in production, these would be real audio files
  const getAudioUrl = (track: AudioTrack) => {
    return `/api/audio/${track.type}/${track.title.toLowerCase().replace(/\s+/g, "-")}.mp3`
  }

  const getTrackColor = (type: string) => {
    switch (type) {
      case "grounding":
      case "anxiety":
        return "bg-red-100 text-red-800"
      case "mood":
        return "bg-blue-100 text-blue-800"
      case "sleep":
        return "bg-purple-100 text-purple-800"
      case "mindfulness":
        return "bg-green-100 text-green-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const handlePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleRestart = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0
      setProgress(0)
      setCurrentTime("0:00")
    }
  }

  const handleTrackSelect = (index: number) => {
    setCurrentTrack(index)
    setIsPlaying(false)
    setProgress(0)
    setCurrentTime("0:00")
  }

  // Simulate audio progress for demo
  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isPlaying) {
      interval = setInterval(() => {
        setProgress((prev) => {
          const newProgress = prev + 1
          if (newProgress >= 100) {
            setIsPlaying(false)
            return 0
          }
          return newProgress
        })

        // Update current time display
        const totalSeconds = Math.floor((progress / 100) * 180) // Assume 3 min tracks
        const minutes = Math.floor(totalSeconds / 60)
        const seconds = totalSeconds % 60
        setCurrentTime(`${minutes}:${seconds.toString().padStart(2, "0")}`)
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [isPlaying, progress])

  return (
    <div className="space-y-4">
      {/* Track List */}
      <div className="grid gap-3">
        {tracks.map((track, index) => (
          <Card
            key={index}
            className={`cursor-pointer transition-all ${
              currentTrack === index ? "ring-2 ring-blue-500 bg-blue-50" : "hover:bg-gray-50"
            }`}
            onClick={() => handleTrackSelect(index)}
          >
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                    <Volume2 className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-900">{track.title}</h4>
                    <p className="text-sm text-gray-600">{track.duration}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={getTrackColor(track.type)}>{track.type}</Badge>
                  {currentTrack === index && <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Audio Player Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="text-center">{tracks[currentTrack]?.title || "Select a track"}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress Bar */}
          <div className="space-y-2">
            <Progress value={progress} className="h-2" />
            <div className="flex justify-between text-sm text-gray-500">
              <span>{currentTime}</span>
              <span>{tracks[currentTrack]?.duration || "0:00"}</span>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="flex items-center justify-center gap-4">
            <Button variant="outline" size="icon" onClick={handleRestart} disabled={!tracks[currentTrack]}>
              <RotateCcw className="w-4 h-4" />
            </Button>

            <Button
              size="lg"
              onClick={handlePlayPause}
              disabled={!tracks[currentTrack]}
              className="w-16 h-16 rounded-full"
            >
              {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6 ml-1" />}
            </Button>

            <Button variant="outline" size="icon" disabled={!tracks[currentTrack]}>
              <Download className="w-4 h-4" />
            </Button>
          </div>

          {/* Track Info */}
          {tracks[currentTrack] && (
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <Badge className={getTrackColor(tracks[currentTrack].type)}>
                {tracks[currentTrack].type.charAt(0).toUpperCase() + tracks[currentTrack].type.slice(1)} Therapy
              </Badge>
              <p className="text-sm text-gray-600 mt-2">{getTrackDescription(tracks[currentTrack].type)}</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Hidden audio element for actual playback */}
      <audio
        ref={audioRef}
        src={tracks[currentTrack] ? getAudioUrl(tracks[currentTrack]) : ""}
        onEnded={() => setIsPlaying(false)}
      />
    </div>
  )
}

function getTrackDescription(type: string): string {
  switch (type) {
    case "grounding":
      return "Helps you feel more present and connected to your immediate environment"
    case "anxiety":
      return "Designed to reduce anxiety symptoms and promote calm"
    case "mood":
      return "Uplifting content to help improve your emotional state"
    case "sleep":
      return "Relaxing audio to prepare your mind and body for rest"
    case "mindfulness":
      return "Develops awareness and acceptance of the present moment"
    case "energy":
      return "Motivating content to boost your energy and motivation"
    default:
      return "Therapeutic audio content for mental wellness"
  }
}
