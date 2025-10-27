"use client"

import React, { useState, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import Webcam from 'react-webcam'
import { 
  Brain, Camera, BarChart3, CheckCircle, AlertTriangle,
  TrendingUp, TrendingDown, Eye, Timer, Zap, Activity
} from 'lucide-react'

interface PHQ9Response {
  question: string
  score: number
  confidence: number
  reasoning: string
}

interface AutoFillData {
  timestamp: string
  estimated_total_score: number
  severity_level: string
  confidence: number
  responses: Record<string, { score: number; confidence: number }>
  reasoning: Record<string, string>
  recommendations: string[]
  facial_analysis_summary: {
    primary_emotion: string
    mood_assessment: string
    sleepiness_level: string
    stress_level: string
    fatigue_detected: boolean
    analysis_quality: number
  }
}

const PHQ9FacialIntegration: React.FC = () => {
  const webcamRef = useRef<Webcam>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [autoFillData, setAutoFillData] = useState<AutoFillData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [permissionGranted, setPermissionGranted] = useState(false)

  const PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself - or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed, or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself"
  ]

  const requestCameraPermission = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      setPermissionGranted(true)
      setError(null)
      stream.getTracks().forEach(track => track.stop())
    } catch (err) {
      setError('Camera permission denied. Please enable camera access to use facial analysis.')
      setPermissionGranted(false)
    }
  }, [])

  const analyzeAndAutoFill = useCallback(async () => {
    if (!webcamRef.current) return

    try {
      setIsAnalyzing(true)
      setError(null)

      const imageSrc = webcamRef.current.getScreenshot()
      if (!imageSrc) {
        setError('Failed to capture image')
        return
      }

      const userId = localStorage.getItem('user_id') || localStorage.getItem('temp_user_id') || 'guest_user'

      const response = await fetch('/api/phq9-integration/auto-fill', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageSrc,
          user_id: userId,
          include_raw_metrics: true,
          include_phq9_estimation: true
        }),
      })

      if (!response.ok) throw new Error('Analysis failed')

      const result = await response.json()
      setAutoFillData(result.data)

    } catch (error) {
      console.error('Error analyzing facial expression:', error)
      setError('Failed to analyze facial expression')
    } finally {
      setIsAnalyzing(false)
    }
  }, [])

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'minimal': return 'text-green-600 bg-green-50'
      case 'mild': return 'text-blue-600 bg-blue-50'
      case 'moderate': return 'text-yellow-600 bg-yellow-50'
      case 'moderately severe': return 'text-orange-600 bg-orange-50'
      case 'severe': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getScoreColor = (score: number) => {
    if (score <= 4) return 'text-green-600'
    if (score <= 9) return 'text-blue-600'
    if (score <= 14) return 'text-yellow-600'
    if (score <= 19) return 'text-orange-600'
    return 'text-red-600'
  }

  if (!permissionGranted) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-6 w-6" />
            <span>PHQ-9 Facial Analysis Integration</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <Camera className="h-4 w-4" />
            <AlertDescription>
              This feature uses AI-powered facial analysis to automatically estimate PHQ-9 depression screening scores.
              The analysis is based on facial expressions, sleepiness indicators, stress levels, and fatigue detection.
            </AlertDescription>
          </Alert>
          
          <div className="text-center space-y-4">
            <div className="space-y-2">
              <p className="font-medium">Camera Access Required</p>
              <p className="text-sm text-gray-600">
                Enable camera access to start AI-powered PHQ-9 analysis
              </p>
            </div>
            
            <Button onClick={requestCameraPermission} className="w-full">
              <Camera className="h-4 w-4 mr-2" />
              Enable Camera Access
            </Button>
          </div>

          {error && (
            <Alert className="border-red-200 bg-red-50">
              <AlertDescription className="text-red-800">
                {error}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="w-full space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Brain className="h-6 w-6" />
              <span>PHQ-9 AI-Powered Analysis</span>
            </div>
            <Button onClick={analyzeAndAutoFill} disabled={isAnalyzing}>
              {isAnalyzing ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Camera className="h-4 w-4 mr-2" />
                  Analyze & Auto-Fill
                </>
              )}
            </Button>
          </CardTitle>
        </CardHeader>
        
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Camera Feed */}
            <div className="space-y-2">
              <div className="relative">
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  className="w-full h-64 object-cover rounded-lg border"
                  videoConstraints={{
                    width: 640,
                    height: 480,
                    facingMode: "user"
                  }}
                />
                
                {isAnalyzing && (
                  <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
                    <div className="text-white text-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-2"></div>
                      <p className="text-sm">Analyzing facial expressions...</p>
                    </div>
                  </div>
                )}
              </div>
              
              <div className="text-xs text-gray-500 text-center">
                Position your face clearly in the camera frame for best results
              </div>
            </div>

            {/* Analysis Instructions */}
            <div className="space-y-4">
              <div className="space-y-3">
                <h3 className="font-semibold">How It Works</h3>
                <div className="space-y-2 text-sm text-gray-600">
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <span>AI analyzes facial expressions for emotional indicators</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <span>Detects sleepiness, fatigue, and stress levels</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <span>Estimates PHQ-9 scores for each question</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                    <span>Provides confidence scores and reasoning</span>
                  </div>
                </div>
              </div>

              <Alert className="border-blue-200 bg-blue-50">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription className="text-blue-800">
                  <strong>Note:</strong> This is an AI estimation tool and should not replace professional medical assessment.
                  Always consult with healthcare providers for accurate diagnosis and treatment.
                </AlertDescription>
              </Alert>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {autoFillData && (
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="questions">PHQ-9 Questions</TabsTrigger>
            <TabsTrigger value="analysis">Facial Analysis</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          </TabsList>
          
          <TabsContent value="overview" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Total Score */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <BarChart3 className="h-5 w-5" />
                    <span>Estimated PHQ-9 Score</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <div className={`text-4xl font-bold mb-2 ${getScoreColor(autoFillData.estimated_total_score)}`}>
                      {autoFillData.estimated_total_score}
                    </div>
                    <div className="text-sm text-gray-600 mb-2">out of 27</div>
                    <Badge className={getSeverityColor(autoFillData.severity_level)}>
                      {autoFillData.severity_level}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              {/* Confidence */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5" />
                    <span>Analysis Confidence</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <div className="text-3xl font-bold mb-2">
                      {Math.round(autoFillData.confidence * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Confidence Level</div>
                    <Progress value={autoFillData.confidence * 100} className="mt-2" />
                  </div>
                </CardContent>
              </Card>

              {/* Analysis Quality */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Activity className="h-5 w-5" />
                    <span>Analysis Quality</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <div className="text-3xl font-bold mb-2">
                      {Math.round(autoFillData.facial_analysis_summary.analysis_quality * 100)}%
                    </div>
                    <div className="text-sm text-gray-600">Image Quality</div>
                    <Progress value={autoFillData.facial_analysis_summary.analysis_quality * 100} className="mt-2" />
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="questions" className="space-y-4">
            <div className="space-y-4">
              {PHQ9_QUESTIONS.map((question, index) => {
                const questionKey = `q${index + 1}_${['interest', 'mood', 'sleep', 'energy', 'appetite', 'self_worth', 'concentration', 'psychomotor', 'suicidal'][index]}`
                const response = autoFillData.responses[questionKey]
                const reasoning = autoFillData.reasoning[questionKey]
                
                if (!response) return null
                
                return (
                  <Card key={index}>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium">
                        Question {index + 1}: {question}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Estimated Score:</span>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-lg px-3 py-1">
                            {response.score}/3
                          </Badge>
                          <span className="text-sm text-gray-500">
                            {Math.round(response.confidence * 100)}% confidence
                          </span>
                        </div>
                      </div>
                      
                      <div className="text-xs text-gray-600 bg-gray-50 p-2 rounded">
                        <strong>AI Reasoning:</strong> {reasoning}
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </TabsContent>

          <TabsContent value="analysis" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Facial Analysis Summary */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Facial Expression Analysis</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span>Primary Emotion:</span>
                    <Badge variant="outline">{autoFillData.facial_analysis_summary.primary_emotion}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Mood Assessment:</span>
                    <Badge variant="outline">{autoFillData.facial_analysis_summary.mood_assessment}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Sleepiness Level:</span>
                    <Badge variant="outline">{autoFillData.facial_analysis_summary.sleepiness_level}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Stress Level:</span>
                    <Badge variant="outline">{autoFillData.facial_analysis_summary.stress_level}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Fatigue Detected:</span>
                    <Badge variant={autoFillData.facial_analysis_summary.fatigue_detected ? "destructive" : "secondary"}>
                      {autoFillData.facial_analysis_summary.fatigue_detected ? "Yes" : "No"}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              {/* Analysis Timestamp */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Analysis Details</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span>Analysis Time:</span>
                    <span className="text-sm">{new Date(autoFillData.timestamp).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Source:</span>
                    <span className="text-sm">{autoFillData.source}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Confidence:</span>
                    <span className="text-sm">{Math.round(autoFillData.confidence * 100)}%</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="recommendations" className="space-y-4">
            <div className="space-y-3">
              {autoFillData.recommendations.map((recommendation, index) => (
                <Alert key={index} className="border-blue-200 bg-blue-50">
                  <TrendingUp className="h-4 w-4" />
                  <AlertDescription className="text-blue-800">
                    {recommendation}
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      )}

      {error && (
        <Alert className="border-red-200 bg-red-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription className="text-red-800">
            {error}
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}

export default PHQ9FacialIntegration
