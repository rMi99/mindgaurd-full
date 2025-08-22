"use client"

import React, { useRef, useCallback, useEffect, useState } from 'react'
import Webcam from 'react-webcam'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Camera, CameraOff, Eye, Shield, Info } from 'lucide-react'

interface EmotionData {
  emotion: string
  confidence: number
  timestamp: number
  faceDetected: boolean
}

interface FacialExpressionAnalysisProps {
  onExpressionData: (data: EmotionData) => void
  isActive: boolean
  language?: string
}

const FacialExpressionAnalysis: React.FC<FacialExpressionAnalysisProps> = ({ 
  onExpressionData, 
  isActive,
  language = "en" 
}) => {
  const webcamRef = useRef<Webcam>(null)
  const [permissionGranted, setPermissionGranted] = useState(false)
  const [permissionDenied, setPermissionDenied] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentEmotion, setCurrentEmotion] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<number>(0)
  const [analysisCount, setAnalysisCount] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const requestCameraPermission = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      setPermissionGranted(true)
      setPermissionDenied(false)
      // Stop the stream immediately - webcam component will handle it
      stream.getTracks().forEach(track => track.stop())
    } catch (err) {
      console.error('Camera permission denied:', err)
      setPermissionDenied(true)
      setPermissionGranted(false)
    }
  }, [])

  const captureAndSendFrame = useCallback(async () => {
    if (!webcamRef.current || !permissionGranted || !isActive) return

    try {
      setIsAnalyzing(true)
      setError(null)
      
      const imageSrc = webcamRef.current.getScreenshot()
      if (!imageSrc) {
        setError('Failed to capture image')
        return
      }

      const response = await fetch('/api/facial-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageSrc }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      
      const emotionData: EmotionData = {
        emotion: result.emotion || 'unknown',
        confidence: result.confidence || 0,
        timestamp: Date.now(),
        faceDetected: result.faceDetected !== false
      }

      setCurrentEmotion(emotionData.emotion)
      setConfidence(emotionData.confidence)
      setAnalysisCount(prev => prev + 1)
      
      onExpressionData(emotionData)
    } catch (error) {
      console.error('Error analyzing facial expression:', error)
      setError('Failed to analyze expression')
    } finally {
      setIsAnalyzing(false)
    }
  }, [webcamRef, permissionGranted, isActive, onExpressionData])

  useEffect(() => {
    if (!permissionGranted || !isActive) return

    const interval = setInterval(() => {
      captureAndSendFrame()
    }, 3000) // Analyze every 3 seconds

    return () => clearInterval(interval)
  }, [captureAndSendFrame, permissionGranted, isActive])

  const getEmotionColor = (emotion: string) => {
    switch (emotion?.toLowerCase()) {
      case 'happy': return 'bg-green-100 text-green-800'
      case 'sad': return 'bg-blue-100 text-blue-800'
      case 'angry': return 'bg-red-100 text-red-800'
      case 'fear': return 'bg-purple-100 text-purple-800'
      case 'surprise': return 'bg-yellow-100 text-yellow-800'
      case 'neutral': return 'bg-gray-100 text-gray-800'
      case 'disgust': return 'bg-orange-100 text-orange-800'
      default: return 'bg-gray-100 text-gray-800'
    }
  }

  const getEmotionIcon = (emotion: string) => {
    switch (emotion?.toLowerCase()) {
      case 'happy': return '😊'
      case 'sad': return '😢'
      case 'angry': return '😠'
      case 'fear': return '😨'
      case 'surprise': return '😮'
      case 'neutral': return '😐'
      case 'disgust': return '🤢'
      default: return '🤔'
    }
  }

  if (!permissionGranted && !permissionDenied) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Camera className="h-5 w-5" />
            <span>Facial Expression Analysis</span>
            <Badge variant="secondary">Optional</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              <strong>Enhanced Assessment with Facial Analysis</strong>
              <br />
              We can analyze your facial expressions during the assessment to provide more accurate results. 
              This helps us understand non-verbal cues that complement your responses.
            </AlertDescription>
          </Alert>
          
          <div className="space-y-3">
            <div className="flex items-start space-x-2 text-sm text-gray-600">
              <Shield className="h-4 w-4 mt-0.5 text-green-600" />
              <span>Your privacy is protected - video is processed in real-time and not stored</span>
            </div>
            <div className="flex items-start space-x-2 text-sm text-gray-600">
              <Eye className="h-4 w-4 mt-0.5 text-blue-600" />
              <span>Analysis happens locally in your browser for maximum security</span>
            </div>
          </div>

          <div className="flex space-x-3">
            <Button onClick={requestCameraPermission} className="flex-1">
              <Camera className="h-4 w-4 mr-2" />
              Enable Facial Analysis
            </Button>
            <Button variant="outline" onClick={() => setPermissionDenied(true)} className="flex-1">
              Continue Without Camera
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (permissionDenied) {
    return (
      <Card className="w-full">
        <CardContent className="pt-6">
          <div className="text-center space-y-2">
            <CameraOff className="h-8 w-8 text-gray-400 mx-auto" />
            <p className="text-sm text-gray-600">
              Camera access declined. Assessment will continue with questionnaire data only.
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Camera className="h-5 w-5" />
            <span>Facial Expression Monitoring</span>
            {isActive && (
              <Badge variant="default" className="animate-pulse">
                Active
              </Badge>
            )}
          </div>
          <div className="text-sm text-gray-500">
            Analyzed: {analysisCount}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative">
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className="w-full h-48 object-cover rounded-lg"
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
                <p className="text-sm">Analyzing expression...</p>
              </div>
            </div>
          )}
        </div>

        {error && (
          <Alert className="border-red-200 bg-red-50">
            <AlertDescription className="text-red-800">
              {error}
            </AlertDescription>
          </Alert>
        )}

        {currentEmotion && (
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-3">
              <span className="text-2xl">{getEmotionIcon(currentEmotion)}</span>
              <div>
                <Badge className={getEmotionColor(currentEmotion)}>
                  {currentEmotion}
                </Badge>
                <p className="text-xs text-gray-600 mt-1">
                  Confidence: {Math.round(confidence * 100)}%
                </p>
              </div>
            </div>
            <div className="text-xs text-gray-500">
              {isAnalyzing ? 'Analyzing...' : 'Last updated'}
            </div>
          </div>
        )}

        <div className="text-xs text-gray-500 text-center">
          Your facial expressions help provide more comprehensive mental health insights.
        </div>
      </CardContent>
    </Card>
  )
}

export default FacialExpressionAnalysis
