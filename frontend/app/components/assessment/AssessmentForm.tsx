"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { ArrowLeft, ArrowRight } from "lucide-react"
import SliderInput from "./SliderInput"
import SleepTracker from "./SleepTracker"
import TextAnalysis from "./TextAnalysis"
import type { AssessmentData, PHQ9Data, SleepData, BehavioralData } from "@/lib/types"

interface AssessmentFormProps {
  onComplete: (data: AssessmentData) => void
  isLoading: boolean
  onBack: () => void
}

export default function AssessmentForm({ onComplete, isLoading, onBack }: AssessmentFormProps) {
  const [currentStep, setCurrentStep] = useState(1)
  const [phq9Data, setPHQ9Data] = useState<PHQ9Data>({
    scores: Array(9).fill(0),
    totalScore: 0,
  })
  const [sleepData, setSleepData] = useState<SleepData>({
    averageHours: 7,
    quality: 3,
    consistency: 3,
    weeklyPattern: Array(7).fill(7),
  })
  const [behavioralData, setBehavioralData] = useState<BehavioralData>({
    moodDescription: "",
    stressLevel: 3,
    socialConnections: 3,
    physicalActivity: 3,
    sentimentScore: 0,
  })

  const totalSteps = 3
  const progress = (currentStep / totalSteps) * 100

  const handleNext = () => {
    if (currentStep < totalSteps) {
      setCurrentStep(currentStep + 1)
    } else {
      handleSubmit()
    }
  }

  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1)
    } else {
      onBack()
    }
  }

  const handleSubmit = () => {
    const assessmentData: AssessmentData = {
      phq9: phq9Data,
      sleep: sleepData,
      behavioral: behavioralData,
      timestamp: new Date().toISOString(),
    }
    onComplete(assessmentData)
  }

  const isStepComplete = () => {
    switch (currentStep) {
      case 1:
        return phq9Data.scores.every((score) => score >= 0)
      case 2:
        return sleepData.averageHours > 0
      case 3:
        return behavioralData.moodDescription.length > 10
      default:
        return false
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Progress Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-2xl font-bold text-gray-900">Mental Health Assessment</h1>
            <span className="text-sm text-gray-500">
              Step {currentStep} of {totalSteps}
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Step Content */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>
              {currentStep === 1 && "PHQ-9 Depression Screening"}
              {currentStep === 2 && "Sleep Pattern Analysis"}
              {currentStep === 3 && "Behavioral & Mood Assessment"}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {currentStep === 1 && (
              <SliderInput
                title="PHQ-9 Depression Scale"
                subtitle="Over the last 2 weeks, how often have you been bothered by any of the following problems?"
                questions={[
                  "Little interest or pleasure in doing things",
                  "Feeling down, depressed, or hopeless",
                  "Trouble falling or staying asleep, or sleeping too much",
                  "Feeling tired or having little energy",
                  "Poor appetite or overeating",
                  "Feeling bad about yourself or that you are a failure",
                  "Trouble concentrating on things",
                  "Moving or speaking slowly, or being fidgety/restless",
                  "Thoughts that you would be better off dead or hurting yourself",
                ]}
                range={[0, 3]}
                labels={["Not at all", "Several days", "More than half the days", "Nearly every day"]}
                values={phq9Data.scores}
                onChange={(scores) => {
                  const totalScore = scores.reduce((sum, score) => sum + score, 0)
                  setPHQ9Data({ scores, totalScore })
                }}
                tooltips={{
                  clinical: "PHQ-9 is a validated depression screening tool used by healthcare professionals worldwide",
                  scoring: "0-4: Minimal, 5-9: Mild, 10-14: Moderate, 15-19: Moderately severe, 20-27: Severe",
                }}
              />
            )}

            {currentStep === 2 && (
              <SleepTracker type="interactive_calendar" data={sleepData} onChange={setSleepData} defaultHours={7} />
            )}

            {currentStep === 3 && (
              <TextAnalysis
                data={behavioralData}
                onChange={setBehavioralData}
                multilingual={true}
                placeholder="Describe your mood and feelings this week... (minimum 10 characters)"
                sentimentAnalysis={true}
              />
            )}
          </CardContent>
        </Card>

        {/* Navigation */}
        <div className="flex justify-between">
          <Button onClick={handlePrevious} variant="outline" className="flex items-center gap-2 bg-transparent">
            <ArrowLeft className="w-4 h-4" />
            {currentStep === 1 ? "Back to Intro" : "Previous"}
          </Button>

          <Button
            onClick={handleNext}
            disabled={!isStepComplete() || isLoading}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700"
          >
            {isLoading ? (
              "Analyzing..."
            ) : currentStep === totalSteps ? (
              "Analyze My Mental Health"
            ) : (
              <>
                Next
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </Button>
        </div>

        {/* Clinical Disclaimer */}
        <div className="mt-8 p-4 bg-gray-50 rounded-lg text-sm text-gray-600">
          <strong>Clinical Disclaimer:</strong> This assessment is for screening purposes only and does not constitute a
          medical diagnosis. Results should not replace professional medical advice, diagnosis, or treatment. Always
          seek advice from qualified healthcare providers.
        </div>
      </div>
    </div>
  )
}
