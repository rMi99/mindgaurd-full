"use client"

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import Webcam from 'react-webcam'
import { 
  Camera, CameraOff, Activity, Brain, Eye, Timer,
  TrendingUp, TrendingDown, AlertTriangle, CheckCircle,
  Pause, Play, BarChart3, LineChart, Zap
} from 'lucide-react'

interface FacialMetrics {
  mood: string
  sleepiness: string
  fatigue: boolean
  stress: string
  phq9Score: number
  confidence: number
  timestamp: number
}

interface RawMetrics {
  eyeAspectRatio: number
  headAngles: {
    pitch: number
    yaw: number
    roll: number
  }
  microExpressions: {
    muscleTension: number
    asymmetry: number
    microMovementFrequency: number
  }
}

interface SessionData {
  sessionId: string | null
  isActive: boolean
  duration: number
  totalFrames: number
  averageQuality: number
}

const RealTimeFacialDashboard: React.FC = () => {
  const webcamRef = useRef<Webcam>(null)
  const [isActive, setIsActive] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentMetrics, setCurrentMetrics] = useState<FacialMetrics | null>(null)
  const [rawMetrics, setRawMetrics] = useState<RawMetrics | null>(null)
  const [sessionData, setSessionData] = useState<SessionData>({
    sessionId: null,
    isActive: false,
    duration: 0,
    totalFrames: 0,
    averageQuality: 0
  })
  const [permissionGranted, setPermissionGranted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [historicalData, setHistoricalData] = useState<FacialMetrics[]>([])
  const [insights, setInsights] = useState<string[]>([])
  const [recommendations, setRecommendations] = useState<string[]>([])

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

  const startSession = useCallback(async () => {
    try {
      const userId = localStorage.getItem('user_id') || localStorage.getItem('temp_user_id') || 'guest_user'
      
      const response = await fetch('/api/facial-dashboard/session/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_id: userId }),
      })

      if (!response.ok) throw new Error('Failed to start session')

      const data = await response.json()
      setSessionData(prev => ({
        ...prev,
        sessionId: data.session_id,
        isActive: true
      }))
      setIsActive(true)
      setError(null)
    } catch (error) {
      console.error('Error starting session:', error)
      setError('Failed to start analysis session')
    }
  }, [])

  const stopSession = useCallback(async () => {
    try {
      if (!sessionData.sessionId) return

      const response = await fetch(`/api/facial-dashboard/session/${sessionData.sessionId}/stop`, {
        method: 'POST',
      })

      if (!response.ok) throw new Error('Failed to stop session')

      const data = await response.json()
      setSessionData(prev => ({
        ...prev,
        isActive: false
      }))
      setIsActive(false)
    } catch (error) {
      console.error('Error stopping session:', error)
      setError('Failed to stop analysis session')
    }
  }, [sessionData.sessionId])

  const captureAndAnalyze = useCallback(async () => {
    if (!webcamRef.current || !isActive || !sessionData.sessionId) return

    try {
      setIsAnalyzing(true)
      setError(null)

      const imageSrc = webcamRef.current.getScreenshot()
      if (!imageSrc) {
        setError('Failed to capture image')
        return
      }

      const userId = localStorage.getItem('user_id') || localStorage.getItem('temp_user_id') || 'guest_user'

      const response = await fetch('/api/facial-dashboard/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageSrc,
          session_id: sessionData.sessionId,
          user_id: userId,
          include_raw_metrics: true,
          include_phq9_estimation: true
        }),
      })

      if (!response.ok) throw new Error('Analysis failed')

      const result = await response.json()

      const metrics: FacialMetrics = {
        mood: result.mood_assessment,
        sleepiness: result.sleepiness.level,
        fatigue: result.fatigue.overall_fatigue,
        stress: result.stress.level,
        phq9Score: result.phq9_estimation.estimated_score,
        confidence: result.emotion_confidence,
        timestamp: Date.now()
      }

      const raw: RawMetrics = {
        eyeAspectRatio: result.eye_metrics?.avg_ear || 0,
        headAngles: {
          pitch: result.head_pose?.pitch || 0,
          yaw: result.head_pose?.yaw || 0,
          roll: result.head_pose?.roll || 0
        },
        microExpressions: {
          muscleTension: result.micro_expressions?.muscle_tension || 0,
          asymmetry: result.micro_expressions?.asymmetry || 0,
          microMovementFrequency: result.micro_expressions?.micro_movement_frequency || 0
        }
      }

      setCurrentMetrics(metrics)
      setRawMetrics(raw)
      setHistoricalData(prev => [...prev.slice(-19), metrics]) // Keep last 20 readings

    } catch (error) {
      console.error('Error analyzing facial expression:', error)
      setError('Failed to analyze facial expression')
    } finally {
      setIsAnalyzing(false)
    }
  }, [isActive, sessionData.sessionId])

  // Auto-capture every 3 seconds when active
  useEffect(() => {
    if (!isActive) return

    const interval = setInterval(captureAndAnalyze, 3000)
    return () => clearInterval(interval)
  }, [captureAndAnalyze, isActive])

  // Update session data periodically
  useEffect(() => {
    if (!sessionData.sessionId || !sessionData.isActive) return

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/facial-dashboard/session/${sessionData.sessionId}/status`)
        if (response.ok) {
          const data = await response.json()
          if (data.status === 'active') {
            setSessionData(prev => ({
              ...prev,
              duration: data.duration_seconds,
              totalFrames: data.total_frames_analyzed,
              averageQuality: data.average_quality
            }))
          }
        }
      } catch (error) {
        console.error('Error updating session status:', error)
      }
    }, 5000)

    return () => clearInterval(interval)
  }, [sessionData.sessionId, sessionData.isActive])

  // Load insights and recommendations
  useEffect(() => {
    const loadInsights = async () => {
      try {
        const userId = localStorage.getItem('user_id') || localStorage.getItem('temp_user_id') || 'guest_user'
        const response = await fetch(`/api/facial-dashboard/insights/${userId}?days=7`)
        
        if (response.ok) {
          const data = await response.json()
          setInsights(data.insights || [])
          setRecommendations(data.recommendations || [])
        }
      } catch (error) {
        console.error('Error loading insights:', error)
      }
    }

    loadInsights()
  }, [])

  const getMoodColor = (mood: string) => {
    switch (mood?.toLowerCase()) {
      case 'happy': return 'text-green-600 bg-green-50'
      case 'sad': return 'text-blue-600 bg-blue-50'
      case 'angry': return 'text-red-600 bg-red-50'
      case 'neutral': return 'text-gray-600 bg-gray-50'
      case 'surprised': return 'text-yellow-600 bg-yellow-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getStressColor = (stress: string) => {
    switch (stress?.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-50'
      case 'medium': return 'text-yellow-600 bg-yellow-50'
      case 'high': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const getSleepinessColor = (sleepiness: string) => {
    switch (sleepiness?.toLowerCase()) {
      case 'alert': return 'text-green-600 bg-green-50'
      case 'slightly tired': return 'text-yellow-600 bg-yellow-50'
      case 'very tired': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  if (!permissionGranted) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-6 w-6" />
            <span>Real-Time Facial Assessment Dashboard</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <Camera className="h-4 w-4" />
            <AlertDescription>
              This dashboard provides real-time analysis of your facial expressions to assess mood, 
              sleepiness, fatigue, stress levels, and estimate PHQ-9 scores for comprehensive mental health insights.
            </AlertDescription>
          </Alert>
          
          <div className="text-center space-y-4">
            <div className="space-y-2">
              <p className="font-medium">Camera Access Required</p>
              <p className="text-sm text-gray-600">
                Enable camera access to start real-time facial analysis
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
      {/* Header with Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Brain className="h-6 w-6" />
              <span>Real-Time Facial Assessment Dashboard</span>
              {isActive && (
                <Badge variant="default" className="animate-pulse">
                  Live Analysis
                </Badge>
              )}
            </div>
            <div className="flex items-center space-x-2">
              {!isActive ? (
                <Button onClick={startSession} className="flex items-center space-x-2">
                  <Play className="h-4 w-4" />
                  <span>Start Analysis</span>
                </Button>
              ) : (
                <Button onClick={stopSession} variant="outline" className="flex items-center space-x-2">
                  <Pause className="h-4 w-4" />
                  <span>Stop Analysis</span>
                </Button>
              )}
            </div>
          </CardTitle>
        </CardHeader>
        
        {isActive && (
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
                        <p className="text-sm">Analyzing...</p>
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="text-xs text-gray-500 text-center">
                  Analysis updates every 3 seconds
                </div>
              </div>

              {/* Session Info */}
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {formatDuration(sessionData.duration)}
                    </div>
                    <div className="text-sm text-blue-800">Duration</div>
                  </div>
                  
                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {sessionData.totalFrames}
                    </div>
                    <div className="text-sm text-green-800">Analyses</div>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Analysis Quality</span>
                    <span>{Math.round(sessionData.averageQuality * 100)}%</span>
                  </div>
                  <Progress value={sessionData.averageQuality * 100} className="h-2" />
                </div>
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Real-time Metrics */}
      {currentMetrics && (
        <Tabs defaultValue="processed" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="processed">Processed Results</TabsTrigger>
            <TabsTrigger value="raw">Raw Metrics</TabsTrigger>
            <TabsTrigger value="trends">Trends</TabsTrigger>
          </TabsList>
          
          <TabsContent value="processed" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Mood */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <span className="text-2xl">😊</span>
                    <span>Current Mood</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className={`text-center py-2 px-3 rounded-lg ${getMoodColor(currentMetrics.mood)}`}>
                    <div className="font-semibold">{currentMetrics.mood}</div>
                    <div className="text-xs opacity-75">
                      {Math.round(currentMetrics.confidence * 100)}% confidence
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Sleepiness */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Eye className="h-5 w-5" />
                    <span>Alertness</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className={`text-center py-2 px-3 rounded-lg ${getSleepinessColor(currentMetrics.sleepiness)}`}>
                    <div className="font-semibold">{currentMetrics.sleepiness}</div>
                    <div className="text-xs opacity-75">Sleep Status</div>
                  </div>
                </CardContent>
              </Card>

              {/* Fatigue */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Timer className="h-5 w-5" />
                    <span>Fatigue</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className={`text-center py-2 px-3 rounded-lg ${
                    currentMetrics.fatigue ? 'text-red-600 bg-red-50' : 'text-green-600 bg-green-50'
                  }`}>
                    <div className="font-semibold">
                      {currentMetrics.fatigue ? 'Detected' : 'Not Detected'}
                    </div>
                    <div className="text-xs opacity-75">Fatigue Signs</div>
                  </div>
                </CardContent>
              </Card>

              {/* Stress */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Zap className="h-5 w-5" />
                    <span>Stress Level</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className={`text-center py-2 px-3 rounded-lg ${getStressColor(currentMetrics.stress)}`}>
                    <div className="font-semibold">{currentMetrics.stress}</div>
                    <div className="text-xs opacity-75">Stress Indicators</div>
                  </div>
                </CardContent>
              </Card>

              {/* PHQ-9 Score */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <BarChart3 className="h-5 w-5" />
                    <span>PHQ-9 Score</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <div className="text-3xl font-bold mb-1">{currentMetrics.phq9Score}</div>
                    <div className="text-sm text-gray-600">out of 27</div>
                    <div className="text-xs mt-1">
                      {currentMetrics.phq9Score <= 4 ? 'Minimal' :
                       currentMetrics.phq9Score <= 9 ? 'Mild' :
                       currentMetrics.phq9Score <= 14 ? 'Moderate' :
                       currentMetrics.phq9Score <= 19 ? 'Moderately Severe' : 'Severe'}
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Last Updated */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Activity className="h-5 w-5" />
                    <span>Last Update</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center">
                    <div className="font-semibold">
                      {new Date(currentMetrics.timestamp).toLocaleTimeString()}
                    </div>
                    <div className="text-xs text-gray-600">
                      {Math.round((Date.now() - currentMetrics.timestamp) / 1000)}s ago
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="raw" className="space-y-4">
            {rawMetrics && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Eye Metrics */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Eye Aspect Ratio</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Current EAR:</span>
                        <span className="font-mono">{rawMetrics.eyeAspectRatio.toFixed(3)}</span>
                      </div>
                      <Progress 
                        value={Math.min(100, rawMetrics.eyeAspectRatio * 200)} 
                        className="h-2"
                      />
                      <div className="text-xs text-gray-500">
                        Normal range: 0.25 - 0.35
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Head Pose */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Head Pose Angles</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Pitch:</span>
                        <span className="font-mono">{rawMetrics.headAngles.pitch.toFixed(1)}°</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Yaw:</span>
                        <span className="font-mono">{rawMetrics.headAngles.yaw.toFixed(1)}°</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Roll:</span>
                        <span className="font-mono">{rawMetrics.headAngles.roll.toFixed(1)}°</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Micro-expressions */}
                <Card className="md:col-span-2">
                  <CardHeader>
                    <CardTitle className="text-base">Micro-Expression Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center">
                        <div className="text-sm text-gray-600 mb-1">Muscle Tension</div>
                        <div className="text-lg font-semibold">
                          {Math.round(rawMetrics.microExpressions.muscleTension * 100)}%
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-gray-600 mb-1">Asymmetry</div>
                        <div className="text-lg font-semibold">
                          {Math.round(rawMetrics.microExpressions.asymmetry * 100)}%
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-gray-600 mb-1">Micro-movements</div>
                        <div className="text-lg font-semibold">
                          {rawMetrics.microExpressions.microMovementFrequency.toFixed(2)}/s
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>

          <TabsContent value="trends" className="space-y-4">
            {historicalData.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">PHQ-9 Score Trend</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-32 flex items-end space-x-1">
                      {historicalData.slice(-10).map((data, index) => (
                        <div
                          key={index}
                          className="bg-blue-500 rounded-t flex-1"
                          style={{ height: `${(data.phq9Score / 27) * 100}%` }}
                          title={`PHQ-9: ${data.phq9Score}`}
                        />
                      ))}
                    </div>
                    <div className="text-xs text-gray-500 mt-2">
                      Last 10 readings
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Mood Distribution</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {['Happy', 'Neutral', 'Sad', 'Angry'].map(mood => {
                        const count = historicalData.filter(d => d.mood === mood).length
                        const percentage = historicalData.length > 0 ? (count / historicalData.length) * 100 : 0
                        return (
                          <div key={mood} className="flex items-center space-x-2">
                            <span className="text-sm w-16">{mood}</span>
                            <Progress value={percentage} className="flex-1 h-2" />
                            <span className="text-xs text-gray-500 w-10">
                              {Math.round(percentage)}%
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center text-gray-500">
                    <LineChart className="h-12 w-12 mx-auto mb-2 opacity-50" />
                    <p>Start analysis to view trends</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      )}

      {/* Insights and Recommendations */}
      {(insights.length > 0 || recommendations.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {insights.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <TrendingUp className="h-5 w-5" />
                  <span>Insights</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {insights.map((insight, index) => (
                    <Alert key={index}>
                      <CheckCircle className="h-4 w-4" />
                      <AlertDescription>{insight}</AlertDescription>
                    </Alert>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {recommendations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <AlertTriangle className="h-5 w-5" />
                  <span>Recommendations</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {recommendations.map((recommendation, index) => (
                    <Alert key={index} className="border-blue-200 bg-blue-50">
                      <AlertDescription className="text-blue-800">
                        {recommendation}
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
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

export default RealTimeFacialDashboard
