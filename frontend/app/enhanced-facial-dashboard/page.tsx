"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Brain, Camera, BarChart3, TrendingUp, Eye, Timer, 
  Zap, Activity, Shield, Info, ChevronRight, Cpu, Target,
  RefreshCw, Settings, Gauge
} from 'lucide-react'
import Header from '../components/Header'
import Footer from '../components/Footer'
import AdaptiveFacialDashboard from '../components/AdaptiveFacialDashboard'
import type { Language } from '@/lib/types'

const EnhancedFacialDashboardPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* Hero Section */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center space-x-3">
            <Brain className="h-8 w-8 text-blue-600" />
            <h1 className="text-3xl font-bold text-gray-900">
              Adaptive Facial Assessment Dashboard
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Next-generation AI-powered facial expression analysis with adaptive accuracy tuning,
            automatic overfitting detection, and dynamic model switching for optimal performance.
          </p>
        </div>

        {/* Enhanced Features Overview */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <BarChart3 className="h-5 w-5" />
              <span>Advanced AI Features</span>
            </CardTitle>
            <CardDescription>
              Our adaptive AI system continuously monitors and adjusts model performance for optimal accuracy
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Adaptive Accuracy Tuning */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Target className="h-6 w-6 text-purple-500" />
                  <h3 className="font-semibold">Adaptive Accuracy Tuning</h3>
                </div>
                <p className="text-sm text-gray-600">
                  AI model automatically adjusts accuracy thresholds when overfitting (100% accuracy) 
                  is detected using live data variance analysis.
                </p>
                <div className="text-xs text-purple-600 font-medium">
                  Self-adjusting thresholds prevent overfitting
                </div>
              </div>

              {/* Dynamic Model Switching */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <RefreshCw className="h-6 w-6 text-blue-500" />
                  <h3 className="font-semibold">Dynamic Model Switching</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Factory pattern enables seamless switching between CNN, MobileNet, ResNet, 
                  and other architectures based on performance metrics.
                </p>
                <div className="text-xs text-blue-600 font-medium">
                  Real-time model optimization
                </div>
              </div>

              {/* Strategy Pattern Optimization */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Settings className="h-6 w-6 text-green-500" />
                  <h3 className="font-semibold">Strategy Pattern Optimization</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Multiple optimization strategies including dropout tuning, early stopping, 
                  and adaptive learning rates for optimal model performance.
                </p>
                <div className="text-xs text-green-600 font-medium">
                  Multi-strategy optimization
                </div>
              </div>

              {/* Real-time WebSocket Updates */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Activity className="h-6 w-6 text-orange-500" />
                  <h3 className="font-semibold">Real-time Updates</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Observer pattern enables instant WebSocket-based updates for model changes, 
                  accuracy alerts, and analysis results.
                </p>
                <div className="text-xs text-orange-600 font-medium">
                  Live model monitoring
                </div>
              </div>

              {/* MVC Architecture */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Cpu className="h-6 w-6 text-indigo-500" />
                  <h3 className="font-semibold">MVC Architecture</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Clean separation of concerns with Model-View-Controller pattern for 
                  scalable and maintainable codebase architecture.
                </p>
                <div className="text-xs text-indigo-600 font-medium">
                  Scalable architecture
                </div>
              </div>

              {/* Performance Monitoring */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Gauge className="h-6 w-6 text-red-500" />
                  <h3 className="font-semibold">Performance Monitoring</h3>
                </div>
                <p className="text-sm text-gray-600">
                  Continuous monitoring of model accuracy, variance, and trends with 
                  automatic alerts for performance issues.
                </p>
                <div className="text-xs text-red-600 font-medium">
                  Proactive monitoring
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Design Patterns Implementation */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Cpu className="h-5 w-5" />
              <span>Design Patterns Implementation</span>
            </CardTitle>
            <CardDescription>
              Advanced software architecture patterns for robust and scalable AI systems
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <h3 className="font-semibold text-lg">Backend Patterns</h3>
                <div className="space-y-3">
                  <div className="border-l-4 border-blue-500 pl-4">
                    <h4 className="font-medium">Observer Pattern</h4>
                    <p className="text-sm text-gray-600">
                      Real-time updates between AI inference layer and UI components
                    </p>
                  </div>
                  <div className="border-l-4 border-green-500 pl-4">
                    <h4 className="font-medium">Factory Pattern</h4>
                    <p className="text-sm text-gray-600">
                      Dynamic loading and switching of ML models (CNN, MobileNet, ResNet)
                    </p>
                  </div>
                  <div className="border-l-4 border-purple-500 pl-4">
                    <h4 className="font-medium">Strategy Pattern</h4>
                    <p className="text-sm text-gray-600">
                      Different optimization strategies (dropout tuning, early stopping, adaptive learning rates)
                    </p>
                  </div>
                  <div className="border-l-4 border-orange-500 pl-4">
                    <h4 className="font-medium">MVC Pattern</h4>
                    <p className="text-sm text-gray-600">
                      Scalable backend architecture with clean separation of concerns
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                <h3 className="font-semibold text-lg">Frontend Patterns</h3>
                <div className="space-y-3">
                  <div className="border-l-4 border-blue-500 pl-4">
                    <h4 className="font-medium">Component Architecture</h4>
                    <p className="text-sm text-gray-600">
                      React Hooks for state management with atomic design principles
                    </p>
                  </div>
                  <div className="border-l-4 border-green-500 pl-4">
                    <h4 className="font-medium">WebSocket Integration</h4>
                    <p className="text-sm text-gray-600">
                      Real-time communication with backend for live updates
                    </p>
                  </div>
                  <div className="border-l-4 border-purple-500 pl-4">
                    <h4 className="font-medium">Atomic Design</h4>
                    <p className="text-sm text-gray-600">
                      Reusable UI components (face preview, accuracy meter, logs)
                    </p>
                  </div>
                  <div className="border-l-4 border-orange-500 pl-4">
                    <h4 className="font-medium">State Management</h4>
                    <p className="text-sm text-gray-600">
                      Centralized state management for complex UI interactions
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Technical Stack */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Cpu className="h-5 w-5" />
                <span>AI Model Tech Stack</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-blue-500" />
                <span className="text-sm">TensorFlow / PyTorch for model training and inference</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-blue-500" />
                <span className="text-sm">OpenCV for real-time video stream handling</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-blue-500" />
                <span className="text-sm">Scikit-learn for accuracy monitoring and adaptive correction</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-blue-500" />
                <span className="text-sm">NumPy / Pandas for analytics and data processing</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Shield className="h-5 w-5" />
                <span>Deployment & Security</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-green-500" />
                <span className="text-sm">Docker for containerization and deployment</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-green-500" />
                <span className="text-sm">NGINX as reverse proxy for load balancing</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-green-500" />
                <span className="text-sm">Git + GitHub Actions for CI/CD pipeline</span>
              </div>
              <div className="flex items-start space-x-2">
                <ChevronRight className="h-4 w-4 mt-0.5 text-green-500" />
                <span className="text-sm">AWS S3 + EC2 / Lambda for hosting and model storage</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Important Notice */}
        <Alert className="border-amber-200 bg-amber-50">
          <Info className="h-4 w-4" />
          <AlertDescription className="text-amber-800">
            <strong>Advanced AI Disclaimer:</strong> This adaptive facial analysis system uses advanced AI 
            techniques including automatic model switching and accuracy tuning. While designed for wellness 
            monitoring, it should not be used as a substitute for professional medical diagnosis or treatment. 
            The system continuously adapts to maintain optimal performance and accuracy.
          </AlertDescription>
        </Alert>

        {/* Main Dashboard */}
        <Tabs defaultValue="dashboard" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="dashboard">Adaptive Dashboard</TabsTrigger>
            <TabsTrigger value="architecture">Architecture</TabsTrigger>
            <TabsTrigger value="getting-started">Getting Started</TabsTrigger>
          </TabsList>
          
          <TabsContent value="dashboard" className="space-y-6">
            <AdaptiveFacialDashboard />
          </TabsContent>
          
          <TabsContent value="architecture" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>System Architecture Overview</CardTitle>
                <CardDescription>
                  Understanding the advanced AI architecture and design patterns
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h3 className="font-semibold text-lg">Adaptive AI Pipeline</h3>
                    <div className="space-y-3">
                      <div className="border-l-4 border-blue-500 pl-4">
                        <h4 className="font-medium">Data Capture</h4>
                        <p className="text-sm text-gray-600">
                          Real-time facial capture with quality assessment and preprocessing
                        </p>
                      </div>
                      <div className="border-l-4 border-green-500 pl-4">
                        <h4 className="font-medium">Model Selection</h4>
                        <p className="text-sm text-gray-600">
                          Factory pattern enables dynamic model switching based on performance
                        </p>
                      </div>
                      <div className="border-l-4 border-purple-500 pl-4">
                        <h4 className="font-medium">Adaptive Analysis</h4>
                        <p className="text-sm text-gray-600">
                          AI model with automatic accuracy tuning and overfitting detection
                        </p>
                      </div>
                      <div className="border-l-4 border-orange-500 pl-4">
                        <h4 className="font-medium">Real-time Updates</h4>
                        <p className="text-sm text-gray-600">
                          Observer pattern provides instant WebSocket-based notifications
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-4">
                    <h3 className="font-semibold text-lg">Optimization Strategies</h3>
                    <div className="space-y-3">
                      <div className="border-l-4 border-red-500 pl-4">
                        <h4 className="font-medium">Dropout Tuning</h4>
                        <p className="text-sm text-gray-600">
                          Automatically adjusts dropout rates to prevent overfitting
                        </p>
                      </div>
                      <div className="border-l-4 border-yellow-500 pl-4">
                        <h4 className="font-medium">Early Stopping</h4>
                        <p className="text-sm text-gray-600">
                          Monitors training progress and stops when no improvement
                        </p>
                      </div>
                      <div className="border-l-4 border-indigo-500 pl-4">
                        <h4 className="font-medium">Adaptive Learning Rate</h4>
                        <p className="text-sm text-gray-600">
                          Dynamically adjusts learning rates based on performance
                        </p>
                      </div>
                      <div className="border-l-4 border-pink-500 pl-4">
                        <h4 className="font-medium">Model Switching</h4>
                        <p className="text-sm text-gray-600">
                          Seamlessly switches between different AI architectures
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="font-semibold text-lg mb-4">Adaptive Analysis Pipeline</h3>
                  <div className="flex flex-wrap items-center justify-center space-x-4 space-y-2">
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">1. Capture</div>
                    <ChevronRight className="h-4 w-4" />
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">2. Preprocess</div>
                    <ChevronRight className="h-4 w-4" />
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">3. Model Selection</div>
                    <ChevronRight className="h-4 w-4" />
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">4. Adaptive Analysis</div>
                    <ChevronRight className="h-4 w-4" />
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">5. Accuracy Monitoring</div>
                    <ChevronRight className="h-4 w-4" />
                    <div className="bg-white px-3 py-2 rounded shadow-sm text-sm">6. Real-time Updates</div>
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
                  Get started with adaptive facial assessment in a few simple steps
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
                        <h3 className="font-semibold">Select AI Model</h3>
                        <p className="text-sm text-gray-600">
                          Choose from CNN, MobileNet, or ResNet models. The system will automatically optimize performance.
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
                          Click "Start Analysis" to begin adaptive facial expression monitoring with real-time updates
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-start space-x-3">
                      <div className="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center font-semibold text-sm">
                        4
                      </div>
                      <div>
                        <h3 className="font-semibold">Monitor Performance</h3>
                        <p className="text-sm text-gray-600">
                          Watch the adaptive AI system automatically tune accuracy and switch models for optimal results
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <Alert className="border-blue-200 bg-blue-50">
                  <TrendingUp className="h-4 w-4" />
                  <AlertDescription className="text-blue-800">
                    <strong>Pro Tip:</strong> The adaptive system learns from your usage patterns and automatically 
                    optimizes model performance. For best results, use the dashboard in a well-lit environment 
                    and maintain natural expressions. The system becomes more accurate with longer analysis sessions.
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

export default EnhancedFacialDashboardPage


