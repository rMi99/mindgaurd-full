"use client"

import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import Webcam from 'react-webcam'
import { 
  Camera, CameraOff, Activity, Brain, Eye, Timer,
  TrendingUp, TrendingDown, AlertTriangle, CheckCircle,
  Pause, Play, BarChart3, LineChart, Zap, Settings,
  Cpu, RefreshCw, Target, Gauge
} from 'lucide-react'

interface AdaptiveMetrics {
  mood: string
  sleepiness: string
  fatigue: boolean
  stress: string
  phq9Score: number
  confidence: number
  timestamp: number
  modelType: string
  adaptiveAnalysis: boolean
  modelConfidence: number
}

interface ModelStatus {
  status: string
  model_info: {
    name: string
    accuracy: number
    is_loaded: boolean
  }
  accuracy_stats: {
    current_accuracy: number
    average_accuracy: number
    variance: number
    trend: string
  }
  optimization_strategies: string[]
}

interface WebSocketMessage {
  type: string
  timestamp: string
  data?: any
}

const AdaptiveFacialDashboard: React.FC = () => {
  const webcamRef = useRef<Webcam>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const [isActive, setIsActive] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentMetrics, setCurrentMetrics] = useState<AdaptiveMetrics | null>(null)
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [supportedModels, setSupportedModels] = useState<string[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('cnn')
  const [permissionGranted, setPermissionGranted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [historicalData, setHistoricalData] = useState<AdaptiveMetrics[]>([])
  const [insights, setInsights] = useState<string[]>([])
  const [recommendations, setRecommendations] = useState<string[]>([])
  const [wsConnected, setWsConnected] = useState(false)
  const [accuracyAlerts, setAccuracyAlerts] = useState<any[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      const wsUrl = `ws://localhost:8000/api/facial-analysis/ws/${clientId}`
      
      try {
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws

        ws.onopen = () => {
          setWsConnected(true)
          console.log('WebSocket connected')
          
          // Join session if available
          if (sessionId) {
            ws.send(JSON.stringify({
              type: 'join_session',
              session_id: sessionId
            }))
          }
        }

        ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)
            handleWebSocketMessage(message)
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
          }
        }

        ws.onclose = () => {
          setWsConnected(false)
          console.log('WebSocket disconnected')
          
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000)
        }

        ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          setWsConnected(false)
        }
      } catch (error) {
        console.error('Error connecting to WebSocket:', error)
        setWsConnected(false)
      }
    }

    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [sessionId])

  const handleWebSocketMessage = (message: WebSocketMessage) => {
    switch (message.type) {
      case 'facial_analysis_result':
        if (message.data?.result) {
          const result = message.data.result
          const metrics: AdaptiveMetrics = {
            mood: result.emotion,
            sleepiness: result.sleepiness?.level || result.sleepiness || 'unknown',
            fatigue: result.fatigue?.overall_fatigue || false,
            stress: result.stress?.level || 'unknown',
            phq9Score: result.phq9_estimation?.estimated_score || 0,
            confidence: result.confidence,
            timestamp: Date.now(),
            modelType: result.model_type || 'unknown',
            adaptiveAnalysis: result.adaptive_analysis || false,
            modelConfidence: result.model_confidence || result.confidence
          }
          setCurrentMetrics(metrics)
          setHistoricalData(prev => [...prev.slice(-19), metrics])
        }
        break
      
      case 'model_update':
        if (message.data?.update) {
          console.log('Model update received:', message.data.update)
          loadModelStatus()
        }
        break
      
      case 'accuracy_alert':
        if (message.data?.alert) {
          setAccuracyAlerts(prev => [...prev.slice(-4), {
            ...message.data.alert,
            timestamp: message.timestamp
          }])
        }
        break
      
      case 'session_joined':
        console.log('Joined session:', message.data?.session_id)
        break
      
      default:
        console.log('Unknown WebSocket message type:', message.type)
    }
  }

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
      
      const response = await fetch('/api/facial-analysis/session/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_id: userId }),
      })

      if (!response.ok) throw new Error('Failed to start session')

      const data = await response.json()
      setSessionId(data.session_id)
      setIsActive(true)
      setError(null)
    } catch (error) {
      console.error('Error starting session:', error)
      setError('Failed to start analysis session')
    }
  }, [])

  const stopSession = useCallback(async () => {
    try {
      if (!sessionId) return

      const response = await fetch(`/api/facial-analysis/session/${sessionId}/stop`, {
        method: 'POST',
      })

      if (!response.ok) throw new Error('Failed to stop session')

      setSessionId(null)
      setIsActive(false)
    } catch (error) {
      console.error('Error stopping session:', error)
      setError('Failed to stop analysis session')
    }
  }, [sessionId])

  const switchModel = useCallback(async (modelType: string) => {
    try {
      const response = await fetch('/api/facial-analysis/model/switch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          model_type: modelType,
          user_id: localStorage.getItem('user_id') || 'guest_user'
        }),
      })

      if (!response.ok) throw new Error('Failed to switch model')

      const data = await response.json()
      console.log('Model switched:', data)
      loadModelStatus()
    } catch (error) {
      console.error('Error switching model:', error)
      setError('Failed to switch model')
    }
  }, [])

  const loadModelStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/facial-analysis/model/status')
      if (response.ok) {
        const data = await response.json()
        setModelStatus(data.model_status)
      }
    } catch (error) {
      console.error('Error loading model status:', error)
    }
  }, [])

  const loadSupportedModels = useCallback(async () => {
    try {
      const response = await fetch('/api/facial-analysis/model/supported')
      if (response.ok) {
        const data = await response.json()
        setSupportedModels(data.supported_models)
      }
    } catch (error) {
      console.error('Error loading supported models:', error)
    }
  }, [])

  const captureAndAnalyze = useCallback(async () => {
    if (!webcamRef.current || !isActive || !sessionId) return

    try {
      setIsAnalyzing(true)
      setError(null)

      const imageSrc = webcamRef.current.getScreenshot()
      if (!imageSrc) {
        setError('Failed to capture image')
        return
      }

      const userId = localStorage.getItem('user_id') || localStorage.getItem('temp_user_id') || 'guest_user'

      const response = await fetch('/api/facial-analysis/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageSrc,
          session_id: sessionId,
          user_id: userId,
          include_raw_metrics: true,
          include_phq9_estimation: true
        }),
      })

      if (!response.ok) throw new Error('Analysis failed')

      const result = await response.json()

      const metrics: AdaptiveMetrics = {
        mood: result.emotion,
        sleepiness: result.sleepiness?.level || result.sleepiness || 'unknown',
        fatigue: result.fatigue?.overall_fatigue || false,
        stress: result.stress?.level || 'unknown',
        phq9Score: result.phq9_estimation?.estimated_score || 0,
        confidence: result.confidence,
        timestamp: Date.now(),
        modelType: result.model_type || 'unknown',
        adaptiveAnalysis: result.adaptive_analysis || false,
        modelConfidence: result.model_confidence || result.confidence
      }

      setCurrentMetrics(metrics)
      setHistoricalData(prev => [...prev.slice(-19), metrics])

    } catch (error) {
      console.error('Error analyzing facial expression:', error)
      setError('Failed to analyze facial expression')
    } finally {
      setIsAnalyzing(false)
    }
  }, [isActive, sessionId])

  // Auto-capture every 3 seconds when active
  useEffect(() => {
    if (!isActive) return

    const interval = setInterval(captureAndAnalyze, 3000)
    return () => clearInterval(interval)
  }, [captureAndAnalyze, isActive])

  // Load initial data
  useEffect(() => {
    loadModelStatus()
    loadSupportedModels()
  }, [loadModelStatus, loadSupportedModels])

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
            <span>Adaptive Facial Assessment Dashboard</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <Camera className="h-4 w-4" />
            <AlertDescription>
              This adaptive dashboard provides real-time facial analysis with AI model accuracy tuning,
              automatic overfitting detection, and dynamic model switching for optimal performance.
            </AlertDescription>
          </Alert>
          
          <div className="text-center space-y-4">
            <div className="space-y-2">
              <p className="font-medium">Camera Access Required</p>
              <p className="text-sm text-gray-600">
                Enable camera access to start adaptive facial analysis
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
              <span>Adaptive Facial Assessment Dashboard</span>
              {isActive && (
                <Badge variant="default" className="animate-pulse">
                  Live Analysis
                </Badge>
              )}
              {wsConnected && (
                <Badge variant="secondary" className="bg-green-100 text-green-800">
                  <Activity className="h-3 w-3 mr-1" />
                  Real-time
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
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
                  Adaptive analysis updates every 3 seconds
                </div>
              </div>

              {/* Model Controls */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">AI Model</label>
                  <Select value={selectedModel} onValueChange={(value) => {
                    setSelectedModel(value)
                    switchModel(value)
                  }}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select model" />
                    </SelectTrigger>
                    <SelectContent>
                      {supportedModels.map((model) => (
                        <SelectItem key={model} value={model}>
                          <div className="flex items-center space-x-2">
                            <Cpu className="h-4 w-4" />
                            <span className="capitalize">{model}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {modelStatus && (
                  <div className="space-y-2">
                    <div className="text-sm font-medium">Model Status</div>
                    <div className="text-xs text-gray-600">
                      <div>Current: {modelStatus.model_info?.name || 'Unknown'}</div>
                      <div>Accuracy: {Math.round((modelStatus.model_info?.accuracy || 0) * 100)}%</div>
                      <div>Trend: {modelStatus.accuracy_stats?.trend || 'Unknown'}</div>
                    </div>
                  </div>
                )}
              </div>

              {/* Session Info */}
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-blue-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {sessionId ? 'Active' : 'Inactive'}
                    </div>
                    <div className="text-sm text-blue-800">Session</div>
                  </div>
                  
                  <div className="text-center p-3 bg-green-50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {historicalData.length}
                    </div>
                    <div className="text-sm text-green-800">Analyses</div>
                  </div>
                </div>
                
                {modelStatus?.accuracy_stats && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Model Accuracy</span>
                      <span>{Math.round(modelStatus.accuracy_stats.current_accuracy * 100)}%</span>
                    </div>
                    <Progress value={modelStatus.accuracy_stats.current_accuracy * 100} className="h-2" />
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        )}
      </Card>

      {/* Accuracy Alerts */}
      {accuracyAlerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5" />
              <span>Accuracy Monitoring Alerts</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {accuracyAlerts.map((alert, index) => (
                <Alert key={index} className={
                  alert.status === 'overfitting' ? 'border-red-200 bg-red-50' :
                  alert.status === 'underfitting' ? 'border-yellow-200 bg-yellow-50' :
                  'border-blue-200 bg-blue-50'
                }>
                  <AlertDescription className={
                    alert.status === 'overfitting' ? 'text-red-800' :
                    alert.status === 'underfitting' ? 'text-yellow-800' :
                    'text-blue-800'
                  }>
                    <strong>{alert.status?.toUpperCase()}:</strong> {alert.message}
                    {alert.recommendation && (
                      <div className="text-xs mt-1">
                        Recommendation: {alert.recommendation}
                      </div>
                    )}
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Real-time Metrics */}
      {currentMetrics && (
        <Tabs defaultValue="processed" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="processed">Processed Results</TabsTrigger>
            <TabsTrigger value="model">Model Info</TabsTrigger>
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
                    {currentMetrics.adaptiveAnalysis && (
                      <Badge variant="outline" className="text-xs">
                        <Target className="h-3 w-3 mr-1" />
                        Adaptive
                      </Badge>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className={`text-center py-2 px-3 rounded-lg ${getMoodColor(currentMetrics.mood)}`}>
                    <div className="font-semibold">{currentMetrics.mood}</div>
                    <div className="text-xs opacity-75">
                      {Math.round(currentMetrics.confidence * 100)}% confidence
                    </div>
                    {currentMetrics.modelConfidence !== currentMetrics.confidence && (
                      <div className="text-xs opacity-75 mt-1">
                        Model: {Math.round(currentMetrics.modelConfidence * 100)}%
                      </div>
                    )}
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

              {/* Model Type */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center space-x-2">
                    <Cpu className="h-5 w-5" />
                    <span>AI Model</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center py-2 px-3 rounded-lg bg-purple-50 text-purple-600">
                    <div className="font-semibold capitalize">{currentMetrics.modelType}</div>
                    <div className="text-xs opacity-75">
                      {currentMetrics.adaptiveAnalysis ? 'Adaptive' : 'Standard'}
                    </div>
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

          <TabsContent value="model" className="space-y-4">
            {modelStatus && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Model Information</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Name:</span>
                        <span className="font-mono">{modelStatus.model_info?.name || 'Unknown'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Status:</span>
                        <Badge variant={modelStatus.model_info?.is_loaded ? 'default' : 'secondary'}>
                          {modelStatus.model_info?.is_loaded ? 'Loaded' : 'Not Loaded'}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span>Accuracy:</span>
                        <span className="font-mono">{Math.round((modelStatus.model_info?.accuracy || 0) * 100)}%</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Accuracy Statistics</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Current:</span>
                        <span className="font-mono">{Math.round((modelStatus.accuracy_stats?.current_accuracy || 0) * 100)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Average:</span>
                        <span className="font-mono">{Math.round((modelStatus.accuracy_stats?.average_accuracy || 0) * 100)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Variance:</span>
                        <span className="font-mono">{(modelStatus.accuracy_stats?.variance || 0).toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Trend:</span>
                        <Badge variant={
                          modelStatus.accuracy_stats?.trend === 'improving' ? 'default' :
                          modelStatus.accuracy_stats?.trend === 'declining' ? 'destructive' :
                          'secondary'
                        }>
                          {modelStatus.accuracy_stats?.trend || 'Unknown'}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="md:col-span-2">
                  <CardHeader>
                    <CardTitle className="text-base">Optimization Strategies</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {modelStatus.optimization_strategies?.map((strategy, index) => (
                        <Badge key={index} variant="outline" className="flex items-center space-x-1">
                          <Settings className="h-3 w-3" />
                          <span>{strategy}</span>
                        </Badge>
                      ))}
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
                    <CardTitle className="text-base">Model Usage</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {['CNN', 'MobileNet', 'ResNet'].map(model => {
                        const count = historicalData.filter(d => d.modelType.toLowerCase().includes(model.toLowerCase())).length
                        const percentage = historicalData.length > 0 ? (count / historicalData.length) * 100 : 0
                        return (
                          <div key={model} className="flex items-center space-x-2">
                            <span className="text-sm w-20">{model}</span>
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

export default AdaptiveFacialDashboard
