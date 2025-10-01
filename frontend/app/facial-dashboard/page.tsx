"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Brain, Camera, BarChart3, TrendingUp, Eye, Timer, 
  Zap, Activity, Shield, Info, ChevronRight
} from 'lucide-react'
import Header from '../components/Header'
import Footer from '../components/Footer'
import RealTimeFacialDashboard from '../components/RealTimeFacialDashboard'
import type { Language } from '@/lib/types'

const FacialDashboardPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* Hero Section */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center space-x-3">
            <Brain className="h-8 w-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-900">
              Facial Assessment Dashboard
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Real-time facial expression analysis providing comprehensive insights into mood, 
            sleepiness, fatigue, stress levels, and PHQ-9 mental health scoring.
          </p>
        </div>

        {/* Feature Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Comprehensive Analysis Features</span>
            </CardTitle>
            <CardDescription>
              Our advanced AI system analyzes multiple facial indicators to provide detailed mental health insights
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Mood Analysis */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">😊</span>
                  <h3 className="font-semibold">Mood Detection</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Real-time analysis of facial expressions to identify current emotional state: 
                  Happy, Sad, Angry, Neutral, or Surprised.
                </p>
                <div className="text-xs text-blue-600 font-medium">
                  Output: Emotional state with confidence level
                </div>
              </div>

              {/* Sleepiness Assessment */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Eye className="h-6 w-6 text-blue-500" />
                  <h3 className="font-semibold">Sleepiness Detection</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Eye aspect ratio and blink rate analysis to determine alertness level 
                  and detect signs of tiredness.
                </p>
                <div className="text-xs text-blue-600 font-medium">
                  Output: Alert, Slightly tired, Very tired
                </div>
              </div>

              {/* Fatigue Signs */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Timer className="h-6 w-6 text-orange-500" />
                  <h3 className="font-semibold">Fatigue Indicators</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Detection of yawning patterns and head droop movements that indicate 
                  physical and mental fatigue.
                </p>
                <div className="text-xs text-blue-600 font-medium">
                  Output: Yes/No with confidence score
                </div>
              </div>

              {/* Stress Indicators */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Zap className="h-6 w-6 text-red-500" />
                  <h3 className="font-semibold">Stress Analysis</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Micro-expression analysis including muscle tension and facial asymmetry 
                  to assess current stress levels.
                </p>
                <div className="text-xs text-blue-600 font-medium">
                  Output: Low, Medium, High stress levels
                </div>
              </div>

              {/* PHQ-9 Estimation */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <BarChart3 className="h-6 w-6 text-purple-500" />
                  <h3 className="font-semibold">PHQ-9 Scoring</h3>
                </div>
                <p className="text-sm text-gray-600">
                  AI-powered estimation of PHQ-9 depression screening scores based on 
                  comprehensive facial expression analysis.
                </p>
                <div className="text-xs text-blue-600 font-medium">
                  Output: Score 0–27 with severity classification
                </div>
              </div>

              {/* Real-time Processing */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Activity className="h-6 w-6 text-green-500" />
                  <h3 className="font-semibold">Live Processing</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Continuous analysis with real-time updates, trend tracking, and 
                  session-based insights for comprehensive monitoring.
                </p>
                <div className="text-xs text-blue-600 font-medium">
                  Output: Dynamic dashboard with live metrics
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* System Requirements & Privacy */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Camera className="h-5 w-5" />
                <span>System Requirements</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-blue-500" />
                <span className="text-sm">Camera access for live facial capture</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-blue-500" />
                <span className="text-sm">Modern web browser with WebRTC support</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-blue-500" />
                <span className="text-sm">Good lighting for optimal analysis accuracy</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-blue-500" />
                <span className="text-sm">Stable internet connection for real-time processing</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Shield className="h-5 w-5" />
                <span>Privacy & Security</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-green-500" />
                <span className="text-sm">No video recording - analysis only</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-green-500" />
                <span className="text-sm">Local processing where possible</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-green-500" />
                <span className="text-sm">Encrypted data transmission</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-green-500" />
                <span className="text-sm">Optional data storage for trend analysis</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Important Notice */}
        <Alert className="border-amber-200 bg-amber-50">
          <Info className="h-4 w-4" />
          <AlertDescription className="text-amber-800">
            <strong>Medical Disclaimer:</strong> This facial analysis tool is designed for wellness monitoring 
            and should not be used as a substitute for professional medical diagnosis or treatment. 
            If you have concerns about your mental health, please consult with a qualified healthcare provider.
          </AlertDescription>
        </Alert>

        {/* Main Dashboard */}
        <Tabs defaultValue="dashboard" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="dashboard">Live Dashboard</TabsTrigger>
            <TabsTrigger value="how-it-works">How It Works</TabsTrigger>
            <TabsTrigger value="getting-started">Getting Started</TabsTrigger>
          </TabsList>
          
          <TabsContent value="dashboard" className="space-y-6">
            <RealTimeFacialDashboard />
          </TabsContent>
          
          <TabsContent value="how-it-works" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Technical Overview</CardTitle>
                <CardDescription>
                  Understanding the AI technology behind facial expression analysis
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h3 className="font-semibold text-lg">Data Capture</h3>
                    <div className="space-y-3">
                      <div className="border-l-4 border-blue-500 pl-4">
                        <h4 className="font-medium">Live Video Stream</h4>
                        <p className="text-sm text-gray-600">
                          Continuous capture of facial expressions at 3-second intervals for optimal analysis
                        </p>
                      </div>
                      <div className="border-l-4 border-green-500 pl-4">
                        <h4 className="font-medium">Feature Extraction</h4>
                        <p className="text-sm text-gray-600">
                          Advanced computer vision algorithms identify key facial landmarks and features
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <h3 className="font-semibold text-lg">AI Processing</h3>
                    <div className="space-y-3">
                      <div className="border-l-4 border-purple-500 pl-4">
                        <h4 className="font-medium">Emotion Recognition</h4>
                        <p className="text-sm text-gray-600">
                          Deep learning models trained on diverse datasets classify emotional expressions
                        </p>
                      </div>
                      <div className="border-l-4 border-orange-500 pl-4">
                        <h4 className="font-medium">Physiological Analysis</h4>
                        <p className="text-sm text-gray-600">
                          Eye tracking, head pose estimation, and micro-expression detection for comprehensive assessment
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="font-semibold text-lg mb-4">Analysis Pipeline</h3>
                  <div className="flex flex-wrap items-center justify-center space-x-4 space-y-2">
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">1. Capture</div>
                    <ChevronRight className="h-4 w-4" />
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">2. Detect Face</div>
                    <ChevronRight className="h-4 w-4" />
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">3. Extract Features</div>
                    <ChevronRight className="h-4 w-4" />
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">4. AI Analysis</div>
                    <ChevronRight className="h-4 w-4" />
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">5. Generate Insights</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="getting-started" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Quick Start Guide</CardTitle>
                <CardDescription>
                  Get started with facial assessment in a few simple steps
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-semibold text-sm">
                        1
                      </div>
                      <div>
                        <h3 className="font-semibold">Enable Camera Access</h3>
                        <p className="text-sm text-gray-600">
                          Click "Enable Camera Access" and allow permission when prompted by your browser
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3">
                      <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-semibold text-sm">
                        2
                      </div>
                      <div>
                        <h3 className="font-semibold">Position Yourself</h3>
                        <p className="text-sm text-gray-600">
                          Ensure good lighting and position your face clearly within the camera frame
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-semibold text-sm">
                        3
                      </div>
                      <div>
                        <h3 className="font-semibold">Start Analysis</h3>
                        <p className="text-sm text-gray-600">
                          Click "Start Analysis" to begin real-time facial expression monitoring
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3">
                      <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-semibold text-sm">
                        4
                      </div>
                      <div>
                        <h3 className="font-semibold">View Results</h3>
                        <p className="text-sm text-gray-600">
                          Monitor live metrics, view trends, and get personalized insights and recommendations
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <Alert className="border-blue-200 bg-blue-50">
                  <TrendingUp className="h-4 w-4" />
                  <AlertDescription className="text-blue-800">
                    <strong>Pro Tip:</strong> For best results, use the dashboard in a well-lit environment 
                    and maintain a natural expression. The system becomes more accurate with longer analysis sessions.
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
      
      <Footer language="en" />
    </div>
  )
}

export default FacialDashboardPage
