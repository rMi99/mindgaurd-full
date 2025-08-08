"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Play, Pause, RotateCcw, Flower2 } from "lucide-react"

interface GuidedMeditationProps {
  riskLevel: string
}

export default function GuidedMeditation({ riskLevel }: GuidedMeditationProps) {
  const [isActive, setIsActive] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [progress, setProgress] = useState(0)
  const [timeRemaining, setTimeRemaining] = useState(0)

  const getMeditationSteps = () => {
    switch (riskLevel) {
      case "high":
        return [
          { instruction: "Find a safe, comfortable position", duration: 30 },
          { instruction: "Notice your feet on the ground", duration: 30 },
          { instruction: "Take three deep breaths", duration: 45 },
          { instruction: "Name 5 things you can see", duration: 60 },
          { instruction: "Name 4 things you can touch", duration: 60 },
          { instruction: "Name 3 things you can hear", duration: 60 },
          { instruction: "You are safe in this moment", duration: 45 },
        ]
      case "moderate":
        return [
          { instruction: "Sit comfortably with eyes closed", duration: 30 },
          { instruction: "Focus on your natural breathing", duration: 90 },
          { instruction: "Notice thoughts without judgment", duration: 120 },
          { instruction: "Return attention to your breath", duration: 90 },
          { instruction: "Expand awareness to your body", duration: 90 },
          { instruction: "Send kindness to yourself", duration: 60 },
          { instruction: "Slowly open your eyes", duration: 30 },
        ]
      default:
        return [
          { instruction: "Settle into a comfortable position", duration: 30 },
          { instruction: "Begin with mindful breathing", duration: 120 },
          { instruction: "Scan your body from head to toe", duration: 180 },
          { instruction: "Notice areas of tension", duration: 120 },
          { instruction: "Breathe into tense areas", duration: 120 },
          { instruction: "Cultivate gratitude", duration: 90 },
          { instruction: "Set intention for your day", duration: 60 },
          { instruction: "Return to awareness", duration: 30 },
        ]
    }
  }

  const steps = getMeditationSteps()
  const totalDuration = steps.reduce((sum, step) => sum + step.duration, 0)

  useEffect(() => {
    let interval: NodeJS.Timeout
    if (isActive && currentStep < steps.length) {
      interval = setInterval(() => {
        setTimeRemaining((prev) => {
          if (prev <= 1) {
            if (currentStep < steps.length - 1) {
              setCurrentStep((prev) => prev + 1)
              return steps[currentStep + 1].duration
            } else {
              setIsActive(false)
              return 0
            }
          }
          return prev - 1
        })

        setProgress((prev) => prev + 100 / totalDuration)
      }, 1000)
    }
    return () => clearInterval(interval)
  }, [isActive, currentStep, steps, totalDuration])

  const handleStart = () => {
    if (!isActive && currentStep === 0) {
      setTimeRemaining(steps[0].duration)
    }
    setIsActive(!isActive)
  }

  const handleReset = () => {
    setIsActive(false)
    setCurrentStep(0)
    setProgress(0)
    setTimeRemaining(steps[0].duration)
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const getMeditationType = () => {
    switch (riskLevel) {
      case "high":
        return { name: "Grounding Meditation", color: "bg-red-100 text-red-800" }
      case "moderate":
        return { name: "Mindfulness Meditation", color: "bg-yellow-100 text-yellow-800" }
      default:
        return { name: "Body Scan Meditation", color: "bg-green-100 text-green-800" }
    }
  }

  const meditationType = getMeditationType()

  return (
    <div className="space-y-6">
      {/* Meditation Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Flower2 className="w-5 h-5 text-purple-600" />
              Guided Meditation
            </CardTitle>
            <Badge className={meditationType.color}>{meditationType.name}</Badge>
          </div>
        </CardHeader>
        <CardContent className="text-center space-y-4">
          <div className="text-sm text-gray-600">
            Total Duration: {formatTime(totalDuration)} • {steps.length} Steps
          </div>
        </CardContent>
      </Card>

      {/* Meditation Player */}
      <Card className="border-2 border-purple-200">
        <CardContent className="p-8 text-center space-y-6">
          {/* Current Instruction */}
          <div className="space-y-4">
            <div className="text-sm text-purple-600 font-medium">
              Step {currentStep + 1} of {steps.length}
            </div>
            <div className="text-2xl font-medium text-gray-900 min-h-[3rem] flex items-center justify-center">
              {steps[currentStep]?.instruction || "Ready to begin"}
            </div>
            {isActive && <div className="text-lg text-purple-600">{formatTime(timeRemaining)}</div>}
          </div>

          {/* Progress */}
          <div className="space-y-2">
            <Progress value={progress} className="h-3" />
            <div className="text-sm text-gray-500">{Math.round(progress)}% Complete</div>
          </div>

          {/* Controls */}
          <div className="flex items-center justify-center gap-4">
            <Button
              variant="outline"
              size="icon"
              onClick={handleReset}
              disabled={!isActive && currentStep === 0 && progress === 0}
            >
              <RotateCcw className="w-4 h-4" />
            </Button>

            <Button
              size="lg"
              onClick={handleStart}
              className="w-20 h-20 rounded-full bg-purple-600 hover:bg-purple-700"
            >
              {isActive ? <Pause className="w-8 h-8" /> : <Play className="w-8 h-8 ml-1" />}
            </Button>
          </div>

          {/* Meditation Tips */}
          <div className="text-sm text-gray-600 max-w-md mx-auto">
            {riskLevel === "high" && (
              <p>
                This grounding meditation helps you feel more present and safe. Focus on the physical sensations and
                your immediate environment.
              </p>
            )}
            {riskLevel === "moderate" && (
              <p>
                Allow thoughts to come and go naturally. There's no need to force anything - simply observe with
                kindness.
              </p>
            )}
            {riskLevel === "low" && (
              <p>Take your time with each step. This practice helps develop body awareness and emotional regulation.</p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Meditation Benefits */}
      <Card>
        <CardHeader>
          <CardTitle>Meditation Benefits</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-4 text-sm">
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <h4 className="font-medium text-blue-900 mb-2">Stress Reduction</h4>
              <p className="text-blue-700">Lowers cortisol levels and activates relaxation response</p>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <h4 className="font-medium text-green-900 mb-2">Emotional Regulation</h4>
              <p className="text-green-700">Improves ability to manage difficult emotions</p>
            </div>
            <div className="text-center p-3 bg-purple-50 rounded-lg">
              <h4 className="font-medium text-purple-900 mb-2">Mental Clarity</h4>
              <p className="text-purple-700">Enhances focus, attention, and decision-making</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
