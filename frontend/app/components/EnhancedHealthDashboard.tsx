"use client"

import React, { useState, useRef, useCallback, useEffect } from 'react'
import Webcam from 'react-webcam'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Camera, 
  CameraOff, 
  Play, 
  Pause, 
  RefreshCw,
  Activity,
  Brain,
  Heart,
  Eye,
  Smile,
  TrendingUp,
  Award,
  Clock,
  Target,
  BarChart3,
  Gamepad2,
  Dumbbell
} from 'lucide-react'

// Remove the old HealthStatus interface since we're using the proper nested structure
// The data now comes from analysisResult.health_recommendations and analysisResult.realtime_monitoring

interface Exercise {
  id: string
  name: string
  type: string
  description: string
  duration_minutes: number
  difficulty: string
  instructions: string[]
  benefits: string[]
  target_conditions: string[]
}

interface Game {
  id: string
  name: string
  category: string
  description: string
  estimated_time: number
  difficulty: string
  instructions: string
  benefits: string[]
  target_mood: string[]
}

interface BiometricIndicators {
  estimated_heart_rate: number
  stress_indicators: number
  fatigue_markers: number
  attention_span: number
  eye_strain: number
}

interface RealtimeMonitoring {
  session_duration: number
  analysis_frequency: number
  trend_data: Array<{
    timestamp: string
    mood_score: number
    stress_level: number
    alertness: number
  }>
  alerts: Array<{
    type: string
    message: string
    severity: string
    timestamp: string
  }>
}

interface EnhancedAnalysisResult {
  // Original analysis data
  emotions: Record<string, number>
  dominant_emotion: string
  confidence: number
  
  // Enhanced health analysis
  health_recommendations: {
    health_status: string // Enum value as string
    mood_category: string // Enum value as string
    lifestyle_tips: string[]
    recommended_exercises: Exercise[]
    suggested_games: Game[]
    progress_metrics?: {
      improvement_percentage: number
      exercises_completed: number
      goals_achieved: number
      streak_days: number
    }
  }
  realtime_monitoring: {
    timestamp: string
    session_id: string
    current_health_status: string
    mood_trend: number[]
    stress_trend: number[]
    brain_activity: {
      stress_level: number
      focus_level: number
      energy_level: number
      emotional_stability: number
      cognitive_load: number
      alertness: number
    }
    active_alerts: string[]
    recommendations_queue: string[]
    session_start_time: string
    total_analysis_frames: number
    average_mood_score: number
  }
  
  // Analysis metadata
  analysis_timestamp: string
  user_id?: string
  session_id: string
  image_quality: number
}

export default function EnhancedHealthDashboard() {
  const webcamRef = useRef<Webcam>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<EnhancedAnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sessionActive, setSessionActive] = useState(false)
  const [completedExercises, setCompletedExercises] = useState<string[]>([])
  const [completedGames, setCompletedGames] = useState<string[]>([])

  const startCamera = useCallback(() => {
    setIsCameraOn(true)
    setError(null)
  }, [])

  const stopCamera = useCallback(() => {
    setIsCameraOn(false)
    setSessionActive(false)
  }, [])

  const startSession = useCallback(() => {
    if (!isCameraOn) {
      setError('Please enable camera first')
      return
    }
    setSessionActive(true)
    setError(null)
  }, [isCameraOn])

  const analyzeFrame = useCallback(async () => {
    if (!webcamRef.current || !sessionActive) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const imageSrc = webcamRef.current.getScreenshot()
      if (!imageSrc) {
        throw new Error('Could not capture image')
      }

      const response = await fetch('/api/facial-dashboard/analyze-enhanced', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageSrc.split(',')[1], // Remove data:image/jpeg;base64, prefix
          session_id: analysisResult?.session_id || `session_${Date.now()}`,
          user_preferences: {
            exercise_difficulty: 'moderate',
            game_categories: ['memory', 'attention'],
            session_duration: 30
          }
        }),
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }

      const result: EnhancedAnalysisResult = await response.json()
      setAnalysisResult(result)
    } catch (err) {
      console.error('Analysis error:', err)
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setIsAnalyzing(false)
    }
  }, [sessionActive, analysisResult?.session_id])

  // Auto-analyze every 5 seconds when session is active
  useEffect(() => {
    if (!sessionActive) return

    const interval = setInterval(analyzeFrame, 5000)
    return () => clearInterval(interval)
  }, [sessionActive, analyzeFrame])

  const completeExercise = useCallback((exerciseId: string) => {
    setCompletedExercises(prev => [...prev, exerciseId])
  }, [])

  const completeGame = useCallback((gameId: string) => {
    setCompletedGames(prev => [...prev, gameId])
  }, [])

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'excellent': return 'bg-green-500'
      case 'good': return 'bg-blue-500'
      case 'moderate': return 'bg-yellow-500'
      case 'concerning': return 'bg-orange-500'
      case 'critical': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  const getMoodColor = (mood: string) => {
    switch (mood.toLowerCase()) {
      case 'positive': return 'text-green-600'
      case 'neutral': return 'text-blue-600'
      case 'stressed': return 'text-orange-600'
      case 'anxious': return 'text-red-600'
      case 'depressed': return 'text-purple-600'
      case 'fatigued': return 'text-gray-600'
      case 'alert': return 'text-emerald-600'
      default: return 'text-gray-600'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">AI Health Dashboard</h1>
          <p className="text-lg text-gray-600">Real-time facial expression analysis and personalized wellness recommendations</p>
        </div>

        {/* Camera Control */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-5 w-5" />
              Camera & Session Control
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col lg:flex-row gap-6">
              <div className="flex-1">
                <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
                  {isCameraOn ? (
                    <Webcam
                      ref={webcamRef}
                      audio={false}
                      screenshotFormat="image/jpeg"
                      className="w-full h-full object-cover"
                      onUserMediaError={(error) => {
                        console.error('Camera error:', error)
                        setError('Camera access denied or not available')
                      }}
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full text-white">
                      <div className="text-center">
                        <CameraOff className="h-16 w-16 mx-auto mb-4 opacity-50" />
                        <p>Camera is disabled</p>
                      </div>
                    </div>
                  )}
                  
                  {sessionActive && (
                    <div className="absolute top-4 left-4">
                      <Badge variant="destructive" className="animate-pulse">
                        <div className="w-2 h-2 bg-white rounded-full mr-2"></div>
                        LIVE
                      </Badge>
                    </div>
                  )}
                </div>
              </div>

              <div className="flex-1 space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <Button
                    onClick={isCameraOn ? stopCamera : startCamera}
                    variant={isCameraOn ? "destructive" : "default"}
                    className="w-full"
                  >
                    {isCameraOn ? (
                      <>
                        <CameraOff className="h-4 w-4 mr-2" />
                        Stop Camera
                      </>
                    ) : (
                      <>
                        <Camera className="h-4 w-4 mr-2" />
                        Start Camera
                      </>
                    )}
                  </Button>

                  <Button
                    onClick={sessionActive ? () => setSessionActive(false) : startSession}
                    variant={sessionActive ? "outline" : "default"}
                    disabled={!isCameraOn}
                    className="w-full"
                  >
                    {sessionActive ? (
                      <>
                        <Pause className="h-4 w-4 mr-2" />
                        End Session
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Start Session
                      </>
                    )}
                  </Button>
                </div>

                <Button
                  onClick={analyzeFrame}
                  disabled={!sessionActive || isAnalyzing}
                  className="w-full"
                  variant="outline"
                >
                  {isAnalyzing ? (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Brain className="h-4 w-4 mr-2" />
                      Analyze Now
                    </>
                  )}
                </Button>

                {error && (
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Analysis Results */}
        {analysisResult && (
          <Tabs defaultValue="overview" className="space-y-6">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="exercises">Exercises</TabsTrigger>
              <TabsTrigger value="games">Games</TabsTrigger>
              <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
              <TabsTrigger value="progress">Progress</TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {/* Health Status */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Activity className="h-5 w-5" />
                      Health Status
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <Badge className={`${getStatusColor(analysisResult?.health_recommendations?.health_status ?? "unknown")} text-white`}>
                        {(analysisResult?.health_recommendations?.health_status ?? "unknown").toUpperCase()}
                      </Badge>
                      <div className={`text-2xl font-bold ${getMoodColor(analysisResult?.health_recommendations?.mood_category ?? "neutral")}`}>
                        {(analysisResult?.health_recommendations?.mood_category ?? "neutral").charAt(0).toUpperCase() +
                         (analysisResult?.health_recommendations?.mood_category ?? "neutral").slice(1)}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Stress Level */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Heart className="h-5 w-5" />
                      Stress Level
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="text-2xl font-bold text-orange-600">
                        {Math.round((analysisResult?.realtime_monitoring?.brain_activity?.stress_level ?? 0) * 100)}%
                      </div>
                      <Progress value={(analysisResult?.realtime_monitoring?.brain_activity?.stress_level ?? 0) * 100} className="w-full" />
                    </div>
                  </CardContent>
                </Card>

                {/* Alertness */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Eye className="h-5 w-5" />
                      Alertness
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="text-2xl font-bold text-blue-600">
                        {Math.round((analysisResult?.realtime_monitoring?.brain_activity?.alertness ?? 0) * 100)}%
                      </div>
                      <Progress value={(analysisResult?.realtime_monitoring?.brain_activity?.alertness ?? 0) * 100} className="w-full" />
                    </div>
                  </CardContent>
                </Card>

                {/* Fatigue Level */}
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Smile className="h-5 w-5" />
                      Energy Level
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="text-2xl font-bold text-green-600">
                        {Math.round((analysisResult?.realtime_monitoring?.brain_activity?.energy_level ?? 0) * 100)}%
                      </div>
                      <Progress value={(analysisResult?.realtime_monitoring?.brain_activity?.energy_level ?? 0) * 100} className="w-full" />
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Recommendations */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    AI Recommendations
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-3">
                    {(analysisResult?.health_recommendations?.lifestyle_tips ?? []).map((rec: string, index: number) => (
                      <div key={index} className="flex items-start gap-3 p-3 bg-blue-50 rounded-lg">
                        <div className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">
                          {index + 1}
                        </div>
                        <p className="text-sm">{rec}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Exercises Tab */}
            <TabsContent value="exercises" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Dumbbell className="h-5 w-5" />
                    Recommended Exercises
                  </CardTitle>
                  <CardDescription>Personalized exercises based on your current state</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4">
                    {analysisResult?.health_recommendations?.recommended_exercises?.map((exercise: Exercise) => (
                      <Card key={exercise.id} className="border-l-4 border-l-blue-500">
                        <CardContent className="pt-6">
                          <div className="flex justify-between items-start mb-4">
                            <div>
                              <h3 className="text-lg font-semibold">{exercise.name}</h3>
                              <div className="flex gap-2 mt-2">
                                <Badge variant="outline">{exercise.type}</Badge>
                                <Badge variant="outline">{exercise.difficulty}</Badge>
                                <Badge variant="outline">{exercise.duration_minutes} min</Badge>
                              </div>
                            </div>
                            <Button
                              onClick={() => completeExercise(exercise.id)}
                              disabled={completedExercises.includes(exercise.id)}
                              variant={completedExercises.includes(exercise.id) ? "outline" : "default"}
                            >
                              {completedExercises.includes(exercise.id) ? (
                                <>
                                  <Award className="h-4 w-4 mr-2" />
                                  Completed
                                </>
                              ) : (
                                'Start Exercise'
                              )}
                            </Button>
                          </div>

                          <div className="space-y-3">
                            <div>
                              <h4 className="font-medium text-sm text-gray-700 mb-2">Instructions:</h4>
                              <ol className="list-decimal list-inside space-y-1 text-sm text-gray-600">
                                {exercise.instructions.map((instruction: string, index: number) => (
                                  <li key={index}>{instruction}</li>
                                ))}
                              </ol>
                            </div>

                            <div>
                              <h4 className="font-medium text-sm text-gray-700 mb-2">Benefits:</h4>
                              <div className="flex flex-wrap gap-1">
                                {exercise.benefits.map((benefit: string, index: number) => (
                                  <Badge key={index} variant="secondary" className="text-xs">
                                    {benefit}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Games Tab */}
            <TabsContent value="games" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Gamepad2 className="h-5 w-5" />
                    Mental Stimulation Games
                  </CardTitle>
                  <CardDescription>Cognitive exercises to enhance mental well-being</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4">
                    {analysisResult?.health_recommendations?.suggested_games?.map((game: Game) => (
                      <Card key={game.id} className="border-l-4 border-l-green-500">
                        <CardContent className="pt-6">
                          <div className="flex justify-between items-start mb-4">
                            <div>
                              <h3 className="text-lg font-semibold">{game.name}</h3>
                              <p className="text-gray-600 mt-1">{game.description}</p>
                              <div className="flex gap-2 mt-2">
                                <Badge variant="outline">{game.category}</Badge>
                                <Badge variant="outline">{game.difficulty}</Badge>
                                <Badge variant="outline">{game.estimated_time} min</Badge>
                              </div>
                            </div>
                            <Button
                              onClick={() => completeGame(game.id)}
                              disabled={completedGames.includes(game.id)}
                              variant={completedGames.includes(game.id) ? "outline" : "default"}
                            >
                              {completedGames.includes(game.id) ? (
                                <>
                                  <Award className="h-4 w-4 mr-2" />
                                  Completed
                                </>
                              ) : (
                                'Play Game'
                              )}
                            </Button>
                          </div>

                          <div>
                            <h4 className="font-medium text-sm text-gray-700 mb-2">Cognitive Benefits:</h4>
                            <div className="flex flex-wrap gap-1">
                              {game.benefits.map((benefit: string, index: number) => (
                                <Badge key={index} variant="secondary" className="text-xs">
                                  {benefit}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Monitoring Tab */}
            <TabsContent value="monitoring" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Biometric Indicators */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Biometric Indicators
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Stress Level</span>
                      <span className="font-bold text-red-600">
                        {Math.round((analysisResult?.realtime_monitoring?.brain_activity?.stress_level ?? 0) * 100)}%
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Focus Level</span>
                      <Progress value={(analysisResult?.realtime_monitoring?.brain_activity?.focus_level ?? 0) * 100} className="w-20" />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Energy Level</span>
                      <Progress value={(analysisResult?.realtime_monitoring?.brain_activity?.energy_level ?? 0) * 100} className="w-20" />
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm">Alertness</span>
                      <Progress value={(analysisResult?.realtime_monitoring?.brain_activity?.alertness ?? 0) * 100} className="w-20" />
                    </div>
                  </CardContent>
                </Card>

                {/* Real-time Alerts */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Clock className="h-5 w-5" />
                      Real-time Alerts
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {analysisResult?.realtime_monitoring?.active_alerts?.length ? (
                        analysisResult.realtime_monitoring.active_alerts.map((alert: string, index: number) => (
                          <Alert key={index} variant="default">
                            <AlertDescription>
                              <div className="flex justify-between items-start">
                                <span>{alert}</span>
                              </div>
                            </AlertDescription>
                          </Alert>
                        ))
                      ) : (
                        <p className="text-gray-500 text-center py-4">No alerts at this time</p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Session Info */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Session Information
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {analysisResult?.realtime_monitoring?.total_analysis_frames ?? 0}
                      </div>
                      <div className="text-sm text-gray-600">Analysis Frames</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {Math.round((analysisResult?.realtime_monitoring?.average_mood_score ?? 0) * 100)}%
                      </div>
                      <div className="text-sm text-gray-600">Avg Mood Score</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {analysisResult?.realtime_monitoring?.mood_trend?.length ?? 0}
                      </div>
                      <div className="text-sm text-gray-600">Mood Trend Points</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">
                        {analysisResult?.realtime_monitoring?.active_alerts?.length ?? 0}
                      </div>
                      <div className="text-sm text-gray-600">Active Alerts</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Progress Tab */}
            <TabsContent value="progress" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Award className="h-5 w-5" />
                    Session Progress
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Session Progress</span>
                        <span className="text-sm text-gray-600">
                          {completedExercises.length + completedGames.length} activities completed
                        </span>
                      </div>
                      <Progress value={(completedExercises.length + completedGames.length) * 20} className="w-full" />
                      <p className="text-center text-2xl font-bold mt-2 text-blue-600">
                        {Math.round((analysisResult?.realtime_monitoring?.average_mood_score ?? 0) * 100)}%
                      </p>
                      <p className="text-center text-sm text-gray-600 mt-1">Average Mood Score</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h3 className="font-medium mb-3">Completed Exercises</h3>
                        <div className="space-y-2">
                          {completedExercises.length > 0 ? (
                            completedExercises.map((exerciseId, index) => (
                              <div key={index} className="flex items-center gap-2 text-sm text-green-600">
                                <Award className="h-4 w-4" />
                                Exercise {index + 1} completed
                              </div>
                            ))
                          ) : (
                            <p className="text-gray-500 text-sm">No exercises completed yet</p>
                          )}
                        </div>
                      </div>

                      <div>
                        <h3 className="font-medium mb-3">Completed Games</h3>
                        <div className="space-y-2">
                          {completedGames.length > 0 ? (
                            completedGames.map((gameId, index) => (
                              <div key={index} className="flex items-center gap-2 text-sm text-green-600">
                                <Award className="h-4 w-4" />
                                Game {index + 1} completed
                              </div>
                            ))
                          ) : (
                            <p className="text-gray-500 text-sm">No games completed yet</p>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}

        {/* Getting Started Message */}
        {!analysisResult && (
          <Card>
            <CardContent className="text-center py-12">
              <Brain className="h-16 w-16 mx-auto mb-4 text-blue-500" />
              <h2 className="text-2xl font-bold mb-2">Ready to Start Your Health Journey?</h2>
              <p className="text-gray-600 mb-6">
                Enable your camera and start a session to receive personalized health recommendations
              </p>
              <div className="flex justify-center gap-4">
                <Button onClick={startCamera} disabled={isCameraOn}>
                  <Camera className="h-4 w-4 mr-2" />
                  Enable Camera
                </Button>
                <Button onClick={startSession} disabled={!isCameraOn} variant="outline">
                  <Play className="h-4 w-4 mr-2" />
                  Start Session
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
