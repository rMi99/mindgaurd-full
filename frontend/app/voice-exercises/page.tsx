"use client"

import React, { useMemo, useState } from "react"
import VoiceMonitor, { type AudioEmotionPoint } from "@/app/components/VoiceMonitor"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

const EXERCISES = [
  {
    id: "breathing",
    title: "Calming Breath + Hum",
    description:
      "Inhale for 4s, exhale for 6s with a soft hum. Repeat for 1 minute. This reduces arousal and relaxes the vagus nerve.",
    script: "Inhale… two… three… four. Exhale with hum… two… three… four… five… six.",
  },
  {
    id: "pitch-glide",
    title: "Pitch Glide (Siren)",
    description:
      "Glide your voice from a comfortable low note to a gentle high note and back. Repeat for 30 seconds to loosen vocal tension.",
    script: "Start low… slowly slide up… and back down… like a soft siren.",
  },
  {
    id: "resonance",
    title: "Resonance Buzz",
    description:
      "Say 'mmm' focusing vibrations around lips and face. Keep airflow easy. Continue for 30 seconds.",
    script: "Mmm… feel a gentle buzz around the lips and cheeks.",
  },
]

export default function VoiceExercisesPage() {
  const [active, setActive] = useState<boolean>(false)
  const [points, setPoints] = useState<AudioEmotionPoint[]>([])
  const [exerciseIndex, setExerciseIndex] = useState(0)

  const summary = useMemo(() => {
    if (points.length === 0) return null
    const aggregate: Record<string, number> = {}
    points.forEach((p) => {
      Object.entries(p.emotions || {}).forEach(([k, v]) => {
        aggregate[k] = (aggregate[k] || 0) + v
      })
    })
    const total = Object.values(aggregate).reduce((a, b) => a + b, 0) || 1
    const normalized = Object.fromEntries(
      Object.entries(aggregate).map(([k, v]) => [k, v / total])
    ) as Record<string, number>
    const dominant = Object.entries(normalized).sort((a, b) => b[1] - a[1])[0][0]
    return { distribution: normalized, dominant, samples: points.length }
  }, [points])

  const current = EXERCISES[exerciseIndex]

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-4">
      <div className="max-w-3xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Voice Exercises</h1>
          <p className="text-gray-600">
            Use these short vocal routines to relax before starting your assessment. We’ll analyze
            tone to provide supportive insights over time.
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>{current.title}</span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => setExerciseIndex((exerciseIndex + EXERCISES.length - 1) % EXERCISES.length)}
                >
                  Prev
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setExerciseIndex((exerciseIndex + 1) % EXERCISES.length)}
                >
                  Next
                </Button>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-gray-700">{current.description}</p>
            <div className="p-3 rounded bg-gray-50 text-sm text-gray-700">{current.script}</div>
            <div>
              <Button onClick={() => setActive((v) => !v)}>
                {active ? "Stop" : "Start"} Exercise
              </Button>
            </div>
            <VoiceMonitor isActive={active} onData={(p) => setPoints((prev) => [...prev, p])} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Session Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">Samples: {points.length}</Badge>
              {summary && (
                <Badge variant="default">Dominant: {summary.dominant}</Badge>
              )}
            </div>
            {summary && (
              <div className="grid grid-cols-2 gap-2 text-sm">
                {Object.entries(summary.distribution).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="capitalize">{k}</span>
                    <span>{Math.round(v * 100)}%</span>
                  </div>
                ))}
              </div>
            )}
            {!summary && <p className="text-sm text-gray-600">No data yet. Start the exercise to see insights.</p>}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}


