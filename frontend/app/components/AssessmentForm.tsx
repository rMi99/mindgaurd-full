"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import type { AssessmentData, DemographicData, PHQ9Data, SleepData, Language, AssessmentResult } from "@/lib/types"
import { getTranslation } from "@/lib/translations"
import { useAuthStore } from "@/lib/stores/authStore"
import DemographicInfo from "./DemographicInfo"
import PHQQuestions from "./PHQQuestions"
import SleepPatterns from "./SleepPatterns"
import Results from "./Results"
import LanguageSelector from "./LanguageSelector"
import CrisisMode from "./CrisisMode"

interface AssessmentFormProps {
  language: Language
  onLanguageChange: (language: Language) => void
  onBack: () => void
}

export default function AssessmentForm({ language, onLanguageChange, onBack }: AssessmentFormProps) {
  const router = useRouter()
  const { user, isAuthenticated } = useAuthStore()
  const [currentStep, setCurrentStep] = useState(1)
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState<AssessmentResult | null>(null)
  const [showCrisisMode, setShowCrisisMode] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [demographicData, setDemographicData] = useState<DemographicData>({
    age: "",
    gender: "",
    region: "",
    education: "",
    employmentStatus: "",
  })

  const [phq9Data, setPHQ9Data] = useState<PHQ9Data>({
    1: null,
    2: null,
    3: null,
    4: null,
    5: null,
    6: null,
    7: null,
    8: null,
    9: null,
  })

  const [sleepData, setSleepData] = useState<SleepData>({
    sleepHours: "",
    sleepQuality: "",
    exerciseFrequency: "",
    stressLevel: "",
    socialSupport: "",
    screenTime: "",
  })

  // Pre-fill demographic data with user info if available
  useEffect(() => {
    if (user && isAuthenticated) {
      setDemographicData(prev => ({
        ...prev,
        age: user.age?.toString() || prev.age,
        gender: user.gender || prev.gender,
        // You can add more user data pre-filling here
      }))
    }
  }, [user, isAuthenticated])

  const totalSteps = 3

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

  const handleSubmit = async () => {
    setIsProcessing(true)
    setError(null)

    // Validate all required data before submission
    const validationErrors = []

    // Check demographics
    if (!demographicData.age || !demographicData.gender) {
      validationErrors.push("Age and gender are required")
    }

    // Check PHQ-9 - ensure all questions are answered
    const phq9Answers = Object.values(phq9Data).filter(v => v !== null && v !== undefined)
    if (phq9Answers.length < 9) {
      validationErrors.push("All PHQ-9 questions must be answered")
    }

    // Check sleep data
    const requiredSleepFields = ["sleepHours", "sleepQuality", "exerciseFrequency", "stressLevel"]
    const missingSleepFields = requiredSleepFields.filter(field => !sleepData[field])
    if (missingSleepFields.length > 0) {
      validationErrors.push(`Missing sleep data: ${missingSleepFields.join(", ")}`)
    }

    if (validationErrors.length > 0) {
      setError(`Please complete all required fields: ${validationErrors.join("; ")}`)
      setIsProcessing(false)
      return
    }

    // Check authentication
    if (!isAuthenticated) {
      localStorage.setItem('redirect_after_login', '/assessment')
      router.push('/auth')
      return
    }

    const assessmentData: AssessmentData = {
      demographics: demographicData,
      phq9: phq9Data,
      sleep: sleepData,
      language,
    }

    try {
      const token = localStorage.getItem('mindguard_token') || localStorage.getItem('access_token')
      if (!token) {
        throw new Error("No authentication token found")
      }

      const response = await fetch("/api/assessment", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(assessmentData),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Assessment failed")
      }

      const result: AssessmentResult = await response.json()
      setResults(result)

      // Save assessment to user history
      const userId = user?.id || localStorage.getItem("mindguard_user_id") || "user_" + Math.random().toString(36).substr(2, 9)
      localStorage.setItem("mindguard_user_id", userId)

      await fetch("/api/dashboard", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          userId,
          assessment: {
            phq9Score: result.phq9Score,
            riskLevel: result.riskLevel,
            sleepData: sleepData,
          },
        }),
      })

      // Check for crisis mode
      if (phq9Data[9] && phq9Data[9] > 0) {
        setShowCrisisMode(true)
        return
      }
    } catch (error: any) {
      console.error("Assessment error:", error)
      setError(error.message || "Failed to submit assessment. Please try again.")
    } finally {
      setIsProcessing(false)
    }
  }

  const isStepComplete = (step: number): boolean => {
    switch (step) {
      case 1:
        return Object.values(demographicData).every((value) => value !== "")
      case 2:
        return Object.values(phq9Data).every((value) => value !== null)
      case 3:
        return Object.values(sleepData).every((value) => value !== "")
      default:
        return false
    }
  }

  // Show error message if there's an error
  if (error) {
    return (
      <div className="max-w-2xl mx-auto p-6">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <h3 className="text-red-800 font-medium mb-2">Assessment Error</h3>
          <p className="text-red-700">{error}</p>
          <button
            onClick={() => setError(null)}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  if (showCrisisMode) {
    return <CrisisMode language={language} onExit={() => setShowCrisisMode(false)} />
  }

  if (results) {
    return (
      <Results
        results={results}
        language={language}
        onStartNew={() => {
          setResults(null)
          setCurrentStep(1)
          setDemographicData({
            age: "",
            gender: "",
            region: "",
            education: "",
            employmentStatus: "",
          })
          setPHQ9Data({
            1: null,
            2: null,
            3: null,
            4: null,
            5: null,
            6: null,
            7: null,
            8: null,
            9: null,
          })
          setSleepData({
            sleepHours: "",
            sleepQuality: "",
            exerciseFrequency: "",
            stressLevel: "",
            socialSupport: "",
            screenTime: "",
          })
        }}
      />
    )
  }

  if (isProcessing) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-lg text-gray-600">{getTranslation(language, "processing")}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-8">
        <LanguageSelector currentLanguage={language} onLanguageChange={onLanguageChange} />
      </div>

      {/* Progress Bar */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">
            {getTranslation(language, "step")} {currentStep} {getTranslation(language, "of")} {totalSteps}
          </span>
          <span className="text-sm text-gray-500">
            {Math.round((currentStep / totalSteps) * 100)}% {getTranslation(language, "complete")}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${(currentStep / totalSteps) * 100}%` }}
          ></div>
        </div>
      </div>

      {/* Step Content */}
      <div className="bg-white rounded-lg shadow-lg p-6 md:p-8">
        {currentStep === 1 && (
          <DemographicInfo data={demographicData} onChange={setDemographicData} language={language} />
        )}

        {currentStep === 2 && <PHQQuestions data={phq9Data} onChange={setPHQ9Data} language={language} />}

        {currentStep === 3 && <SleepPatterns data={sleepData} onChange={setSleepData} language={language} />}

        {/* Navigation Buttons */}
        <div className="flex justify-between mt-8 pt-6 border-t">
          <button
            onClick={handlePrevious}
            className="px-6 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 transition-colors duration-200"
          >
            {getTranslation(language, "previous")}
          </button>

          <button
            onClick={handleNext}
            disabled={!isStepComplete(currentStep)}
            className={`px-6 py-2 rounded-lg font-medium transition-colors duration-200 ${
              isStepComplete(currentStep)
                ? "bg-blue-600 hover:bg-blue-700 text-white"
                : "bg-gray-300 text-gray-500 cursor-not-allowed"
            }`}
          >
            {currentStep === totalSteps
              ? getTranslation(language, "completeAssessment")
              : getTranslation(language, "next")}
          </button>
        </div>
      </div>
    </div>
  )
}
