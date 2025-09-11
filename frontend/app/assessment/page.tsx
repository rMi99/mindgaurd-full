'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Checkbox } from '@/components/ui/checkbox';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { ArrowLeft, ArrowRight, CheckCircle, Brain, Heart, Activity, Users, Moon, Utensils, Camera, Download, AlertTriangle, TrendingUp, Shield } from 'lucide-react';
import { useAssessmentStore } from '@/lib/stores/assessmentStore';
import { useAuthStore } from '@/lib/stores/authStore';
import { useToast } from '@/hooks/use-toast';
import { Toaster } from '@/components/ui/toaster';
import Header from '@/app/components/Header';
import Footer from '@/app/components/Footer';
import FacialExpressionAnalysis from '@/app/components/FacialExpressionAnalysis';

interface AssessmentStep {
  id: number;
  title: string;
  description: string;
  icon: React.ReactNode;
  isCompleted: boolean;
}

interface EmotionData {
  emotion: string;
  confidence: number;
  timestamp: number;
  faceDetected: boolean;
}

interface DetectedMetrics {
  emotion: string;
  sleepLevel: number;
  stressLevel: number;
  fatigueLevel: number;
  alertnessLevel: number;
  confidence: number;
}

interface AssessmentResults {
  mentalHealthStatus: string;
  riskFactors: string[];
  recommendations: string[];
  exercises: Array<{
    name: string;
    description: string;
    duration: string;
    benefits: string[];
  }>;
  futureRiskAssessment: {
    timeframe: string;
    conditions: string[];
    probability: number;
  };
}

const ASSESSMENT_STEPS: AssessmentStep[] = [
  {
    id: 1,
    title: 'Basic Information',
    description: 'Personal details and demographics',
    icon: <Users className="h-6 w-6" />,
    isCompleted: false
  },
  {
    id: 2,
    title: 'Physical Health',
    description: 'Sleep, exercise, and diet patterns',
    icon: <Activity className="h-6 w-6" />,
    isCompleted: false
  },
  {
    id: 3,
    title: 'Mental Wellbeing',
    description: 'Stress levels and emotional health',
    icon: <Brain className="h-6 w-6" />,
    isCompleted: false
  },
  {
    id: 4,
    title: 'Lifestyle Factors',
    description: 'Social connections and work-life balance',
    icon: <Heart className="h-6 w-6" />,
    isCompleted: false
  },
  {
    id: 5,
    title: 'Review & Submit',
    description: 'Review your responses and submit assessment',
    icon: <CheckCircle className="h-6 w-6" />,
    isCompleted: false
  }
];

export default function AssessmentPage() {
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [emotionData, setEmotionData] = useState<EmotionData[]>([]);
  const [facialAnalysisEnabled, setFacialAnalysisEnabled] = useState(false);
  const [detectedMetrics, setDetectedMetrics] = useState<DetectedMetrics | null>(null);
  const [assessmentResults, setAssessmentResults] = useState<AssessmentResults | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [cameraRequested, setCameraRequested] = useState(false);
  
  const { isAuthenticated, user, setIsAuthenticated } = useAuthStore();
  const { toast } = useToast();
  const {
    assessmentData,
    updateAssessmentData,
    validateStep,
    getStepValidation,
    resetAssessment,
    getTransformedData
  } = useAssessmentStore();

  // Check authentication on component mount
  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem('mindguard_token') || localStorage.getItem('access_token');
      if (token) {
        // Set authentication state if token exists
        setIsAuthenticated(true);
      }
      // Always allow access to assessment, just set loading to false
      setIsLoading(false);
    };

    checkAuth();
  }, [setIsAuthenticated]);

  // Request camera access when component mounts
  useEffect(() => {
    const requestCameraOnStart = async () => {
      if (!cameraRequested) {
        setCameraRequested(true);
        try {
          await navigator.mediaDevices.getUserMedia({ video: true });
          setFacialAnalysisEnabled(true);
          toast({
            title: "Camera Access Granted",
            description: "Facial detection will help auto-fill your assessment based on detected emotions and sleep patterns.",
            duration: 5000,
          });
        } catch (error) {
          console.log('Camera access denied or not available');
          toast({
            title: "Camera Access Optional",
            description: "You can still complete the assessment manually. Camera detection helps with more accurate results.",
            variant: "destructive",
            duration: 7000,
          });
        }
      }
    };

    if (!isLoading) {
      requestCameraOnStart();
    }
  }, [isLoading, cameraRequested, toast]);

  const handleEmotionData = (newEmotionData: EmotionData) => {
    setEmotionData(prev => [...prev, newEmotionData]);
    
    // Auto-analyze and update detected metrics
    const recentEmotions = [...emotionData, newEmotionData].slice(-10); // Last 10 readings
    const analysis = analyzeEmotionData(recentEmotions);
    
    if (analysis) {
      const newMetrics: DetectedMetrics = {
        emotion: analysis.dominantEmotion,
        sleepLevel: calculateSleepLevel(analysis.dominantEmotion, analysis.avgConfidence),
        stressLevel: calculateStressLevel(analysis.dominantEmotion, analysis.avgConfidence),
        fatigueLevel: calculateFatigueLevel(analysis.dominantEmotion, analysis.avgConfidence),
        alertnessLevel: calculateAlertnessLevel(analysis.dominantEmotion, analysis.avgConfidence),
        confidence: analysis.avgConfidence
      };
      
      setDetectedMetrics(newMetrics);
      
      // Auto-fill assessment data based on detected metrics
      updateAssessmentData('detectedEmotion', newMetrics.emotion);
      updateAssessmentData('detectedSleepLevel', newMetrics.sleepLevel);
      updateAssessmentData('stressLevel', Math.round(newMetrics.stressLevel * 10));
      
      // Show toast with detected results
      toast({
        title: "Facial Analysis Complete",
        description: `Detected: ${newMetrics.emotion} (${Math.round(newMetrics.confidence * 100)}% confidence). Sleep level: ${newMetrics.sleepLevel}/10, Stress: ${Math.round(newMetrics.stressLevel * 10)}/10`,
        duration: 8000,
      });
    }
  };

  // Helper functions for calculating metrics from emotion data
  const calculateSleepLevel = (emotion: string, confidence: number): number => {
    const sleepMapping: Record<string, number> = {
      'tired': 3, 'sad': 4, 'neutral': 6, 'happy': 8, 'alert': 9, 'angry': 5, 'surprised': 7
    };
    return sleepMapping[emotion] || 6;
  };

  const calculateStressLevel = (emotion: string, confidence: number): number => {
    const stressMapping: Record<string, number> = {
      'angry': 0.9, 'sad': 0.7, 'tired': 0.6, 'neutral': 0.4, 'happy': 0.2, 'alert': 0.3, 'surprised': 0.5
    };
    return stressMapping[emotion] || 0.5;
  };

  const calculateFatigueLevel = (emotion: string, confidence: number): number => {
    const fatigueMapping: Record<string, number> = {
      'tired': 0.9, 'sad': 0.7, 'neutral': 0.5, 'angry': 0.6, 'happy': 0.2, 'alert': 0.1, 'surprised': 0.3
    };
    return fatigueMapping[emotion] || 0.5;
  };

  const calculateAlertnessLevel = (emotion: string, confidence: number): number => {
    const alertnessMapping: Record<string, number> = {
      'alert': 0.9, 'surprised': 0.8, 'happy': 0.7, 'neutral': 0.5, 'angry': 0.6, 'sad': 0.3, 'tired': 0.1
    };
    return alertnessMapping[emotion] || 0.5;
  };

  // Update progress based on completed steps
  const progress = (currentStep / ASSESSMENT_STEPS.length) * 100;

  // Handle next step
  const handleNext = () => {
    const isValid = validateStep(currentStep);
    if (isValid && currentStep < ASSESSMENT_STEPS.length) {
      setCurrentStep(currentStep + 1);
    }
  };

  // Handle previous step
  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  // Analyze emotion data to provide summary
  const analyzeEmotionData = (emotions: EmotionData[]) => {
    if (emotions.length === 0) return null;

    const emotionCounts = emotions.reduce((acc, emotion) => {
      acc[emotion.emotion] = (acc[emotion.emotion] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const dominantEmotion = Object.entries(emotionCounts)
      .sort(([,a], [,b]) => b - a)[0][0];

    const avgConfidence = emotions.reduce((sum, emotion) => sum + emotion.confidence, 0) / emotions.length;

    return {
      dominantEmotion,
      avgConfidence,
      totalSamples: emotions.length,
      emotionDistribution: emotionCounts
    };
  };

  // Handle form submission
  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      const transformedData = getTransformedData();
      
      // Include emotion data if available
      const finalData = {
        ...transformedData,
        emotionAnalysis: emotionData.length > 0 ? {
          data: emotionData,
          summary: analyzeEmotionData(emotionData)
        } : null,
        detectedMetrics: detectedMetrics
      };

      const token = localStorage.getItem('mindguard_token') || localStorage.getItem('access_token');
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }

      const response = await fetch('/api/assessment', {
        method: 'POST',
        headers,
        body: JSON.stringify(finalData),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Assessment submitted successfully:', result);
        
        // Generate assessment results
        const results = generateAssessmentResults(finalData, detectedMetrics);
        setAssessmentResults(results);
        setShowResults(true);
        
        toast({
          title: "Assessment Complete!",
          description: "Your mental health assessment has been processed. View your personalized recommendations below.",
          duration: 5000,
        });
        
        // Reset the assessment store
        resetAssessment();
      } else {
        const error = await response.json();
        console.error('Assessment submission failed:', error);
        toast({
          title: "Submission Failed",
          description: error.detail || 'Unknown error occurred. Please try again.',
          variant: "destructive",
          duration: 5000,
        });
      }
    } catch (error) {
      console.error('Error submitting assessment:', error);
      toast({
        title: "Error",
        description: 'Error submitting assessment. Please check your connection and try again.',
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  // Generate comprehensive assessment results
  const generateAssessmentResults = (data: any, metrics: DetectedMetrics | null): AssessmentResults => {
    const phq9Score = data.phq9 ? Object.values(data.phq9).reduce((sum: number, val: any) => sum + (val || 0), 0) : 0;
    const stressLevel = data.stressLevel || 5;
    
    let mentalHealthStatus = "Healthy";
    let riskLevel = "Low";
    
    if (phq9Score > 14) {
      mentalHealthStatus = "Moderate to Severe Depression Risk";
      riskLevel = "High";
    } else if (phq9Score > 9) {
      mentalHealthStatus = "Mild Depression Risk";
      riskLevel = "Moderate";
    } else if (phq9Score > 4) {
      mentalHealthStatus = "Mild Symptoms";
      riskLevel = "Low-Moderate";
    }

    const riskFactors = [];
    if (phq9Score > 9) riskFactors.push("Depression indicators");
    if (stressLevel > 7) riskFactors.push("High stress levels");
    if (data.sleepHours < 6) riskFactors.push("Sleep deprivation");
    if (data.exerciseFrequency < 2) riskFactors.push("Sedentary lifestyle");
    if (metrics?.fatigueLevel > 0.7) riskFactors.push("High fatigue levels");

    const recommendations = [
      "Regular sleep schedule (7-9 hours)",
      "Daily physical exercise (30+ minutes)",
      "Mindfulness and meditation practice",
      "Social connection and support",
      "Professional counseling if symptoms persist"
    ];

    const exercises = [
      {
        name: "Deep Breathing Exercise",
        description: "Practice 4-7-8 breathing technique to reduce stress and anxiety",
        duration: "5-10 minutes",
        benefits: ["Reduces stress", "Improves focus", "Calms nervous system"]
      },
      {
        name: "Progressive Muscle Relaxation",
        description: "Systematically tense and relax muscle groups",
        duration: "15-20 minutes",
        benefits: ["Reduces physical tension", "Improves sleep", "Decreases anxiety"]
      },
      {
        name: "Mindful Walking",
        description: "Slow, deliberate walking while focusing on sensations",
        duration: "10-30 minutes",
        benefits: ["Combines exercise with mindfulness", "Improves mood", "Increases awareness"]
      },
      {
        name: "Gratitude Journaling",
        description: "Write down 3 things you're grateful for each day",
        duration: "5-10 minutes",
        benefits: ["Improves mood", "Increases positivity", "Enhances life satisfaction"]
      }
    ];

    const futureRiskAssessment = {
      timeframe: riskLevel === "High" ? "1-3 months" : riskLevel === "Moderate" ? "3-6 months" : "6-12 months",
      conditions: riskLevel === "High" ? 
        ["Major Depression", "Anxiety Disorders", "Sleep Disorders"] :
        riskLevel === "Moderate" ? 
          ["Mild Depression", "Stress-related conditions", "Sleep disturbances"] :
          ["General wellness maintenance needed"],
      probability: riskLevel === "High" ? 75 : riskLevel === "Moderate" ? 45 : 15
    };

    return {
      mentalHealthStatus,
      riskFactors,
      recommendations,
      exercises,
      futureRiskAssessment
    };
  };

  const downloadReport = () => {
    if (!assessmentResults) return;
    
    const reportData = {
      assessmentDate: new Date().toLocaleDateString(),
      mentalHealthStatus: assessmentResults.mentalHealthStatus,
      detectedMetrics: detectedMetrics,
      riskFactors: assessmentResults.riskFactors,
      recommendations: assessmentResults.recommendations,
      exercises: assessmentResults.exercises,
      futureRiskAssessment: assessmentResults.futureRiskAssessment
    };
    
    const dataStr = JSON.stringify(reportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `mental-health-assessment-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    toast({
      title: "Report Downloaded",
      description: "Your complete mental health assessment report has been downloaded.",
      duration: 3000,
    });
  };
  
  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading assessment...</p>
        </div>
      </div>
    );
  }

  // Show results page after assessment completion
  if (showResults && assessmentResults) {
    return (
      <>
        <Header />
        <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
          <div className="max-w-6xl mx-auto p-6">
            <div className="text-center mb-8">
              <h1 className="text-4xl font-bold text-gray-900 mb-4">Your Mental Health Assessment Results</h1>
              <p className="text-lg text-gray-600">Comprehensive analysis based on your responses and facial detection data</p>
            </div>

            {/* Mental Health Status */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-6 w-6" />
                  Current Mental Health Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-center">
                  <Badge 
                    variant={assessmentResults.mentalHealthStatus.includes('Severe') ? 'destructive' : 
                             assessmentResults.mentalHealthStatus.includes('Moderate') ? 'secondary' : 'default'}
                    className="text-xl px-6 py-3"
                  >
                    {assessmentResults.mentalHealthStatus}
                  </Badge>
                  {detectedMetrics && (
                    <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-sm text-gray-500">Detected Emotion</div>
                        <div className="font-bold capitalize">{detectedMetrics.emotion}</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-gray-500">Sleep Level</div>
                        <div className="font-bold">{detectedMetrics.sleepLevel}/10</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-gray-500">Stress Level</div>
                        <div className="font-bold">{Math.round(detectedMetrics.stressLevel * 10)}/10</div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-gray-500">Confidence</div>
                        <div className="font-bold">{Math.round(detectedMetrics.confidence * 100)}%</div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Future Risk Assessment */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-6 w-6" />
                  Future Risk Assessment
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className="text-sm text-gray-500 mb-2">Timeframe</div>
                    <Badge variant="outline" className="text-lg px-4 py-2">
                      {assessmentResults.futureRiskAssessment.timeframe}
                    </Badge>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-500 mb-2">Risk Probability</div>
                    <div className="text-3xl font-bold text-orange-600">
                      {assessmentResults.futureRiskAssessment.probability}%
                    </div>
                    <Progress value={assessmentResults.futureRiskAssessment.probability} className="mt-2" />
                  </div>
                  <div>
                    <div className="text-sm text-gray-500 mb-2">Potential Conditions</div>
                    <div className="space-y-1">
                      {assessmentResults.futureRiskAssessment.conditions.map((condition, index) => (
                        <Badge key={index} variant="secondary" className="mr-1 mb-1">
                          {condition}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Risk Factors */}
            {assessmentResults.riskFactors.length > 0 && (
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="h-6 w-6" />
                    Identified Risk Factors
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-3">
                    {assessmentResults.riskFactors.map((factor, index) => (
                      <div key={index} className="flex items-center gap-3 p-3 bg-red-50 rounded-lg">
                        <AlertTriangle className="h-5 w-5 text-red-500" />
                        <span>{factor}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Recommendations */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-6 w-6" />
                  Personalized Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-3">
                  {assessmentResults.recommendations.map((recommendation, index) => (
                    <div key={index} className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                      <CheckCircle className="h-5 w-5 text-green-500" />
                      <span>{recommendation}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Healing Exercises */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-6 w-6" />
                  Recommended Healing Exercises & Practices
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 gap-6">
                  {assessmentResults.exercises.map((exercise, index) => (
                    <Card key={index} className="border-l-4 border-l-blue-500">
                      <CardContent className="pt-6">
                        <h3 className="text-lg font-semibold mb-2">{exercise.name}</h3>
                        <p className="text-gray-600 mb-3">{exercise.description}</p>
                        <div className="flex items-center gap-2 mb-3">
                          <Badge variant="outline">{exercise.duration}</Badge>
                        </div>
                        <div>
                          <div className="text-sm font-medium text-gray-700 mb-2">Benefits:</div>
                          <div className="flex flex-wrap gap-1">
                            {exercise.benefits.map((benefit, idx) => (
                              <Badge key={idx} variant="secondary" className="text-xs">
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

            {/* Action Buttons */}
            <div className="flex justify-center gap-4 mb-8">
              <Button onClick={downloadReport} className="flex items-center gap-2">
                <Download className="h-4 w-4" />
                Download Complete Report
              </Button>
              <Button 
                variant="outline" 
                onClick={() => {
                  setShowResults(false);
                  setCurrentStep(1);
                  setEmotionData([]);
                  setDetectedMetrics(null);
                  setAssessmentResults(null);
                }}
              >
                Take New Assessment
              </Button>
              {isAuthenticated && (
                <Button variant="outline" onClick={() => router.push('/dashboard')}>
                  Go to Dashboard
                </Button>
              )}
            </div>
          </div>
        </div>
        <Footer language="en" />
        <Toaster />
      </>
    );
  }

  // Render component with facial analysis integration
  return (
    <>
      <Header />
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Mental Health Assessment</h1>
          <p className="text-gray-600">
            Step {currentStep} of {ASSESSMENT_STEPS.length}: {ASSESSMENT_STEPS[currentStep - 1]?.title}
          </p>
          <Progress value={progress} className="mt-4 max-w-md mx-auto" />
        </div>

        {/* Authentication Notice */}
        {!isAuthenticated && (
          <Card className="mb-6 border-blue-200 bg-blue-50">
            <CardContent className="pt-6">
              <div className="flex items-center space-x-3">
                <div className="text-blue-600">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="flex-1">
                  <p className="text-sm text-blue-800">
                    <strong>Guest Mode:</strong> You can complete this assessment without creating an account. 
                    However, <a href="/auth" className="underline font-medium">signing up</a> allows you to save results, track progress, and receive personalized recommendations.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Facial and Voice Analysis (only show on Mental Wellbeing step) */}
        {currentStep === 3 && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5" />
                Enhanced Assessment with Facial & Voice Analysis
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-2 mb-4">
                <Checkbox 
                  id="facial-analysis" 
                  checked={facialAnalysisEnabled}
                  onCheckedChange={(checked) => setFacialAnalysisEnabled(checked === true)}
                />
                <Label htmlFor="facial-analysis">
                  Enable facial expression monitoring for more accurate assessment
                </Label>
              </div>
              <p className="text-sm text-gray-600 mb-4">
                This optional feature analyzes your facial expressions during the assessment to provide 
                additional insights. Your privacy is protected - video is processed in real-time and not stored.
              </p>
              
              {facialAnalysisEnabled && (
                <FacialExpressionAnalysis 
                  onExpressionData={handleEmotionData}
                  isActive={facialAnalysisEnabled}
                />
              )}
            </CardContent>
          </Card>
        )}

        {/* Assessment Steps */}
        <Card className="mb-6">
          <CardContent className="p-6">{renderStep()}</CardContent>
        </Card>

        {/* Navigation */}
        <div className="flex justify-between items-center">
          <Button
            variant="outline"
            onClick={handlePrevious}
            disabled={currentStep === 1}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Previous
          </Button>

          <div className="flex items-center gap-2">
            {emotionData.length > 0 && (
              <Badge variant="secondary" className="text-xs">
                {emotionData.length} emotion samples collected
              </Badge>
            )}
            {/* Debug validation status */}
            {process.env.NODE_ENV === 'development' && (
              <Badge variant="outline" className="text-xs">
                Step {currentStep} Valid: {getStepValidation(currentStep).isValid ? 'Yes' : 'No'}
              </Badge>
            )}
          </div>

          {currentStep < ASSESSMENT_STEPS.length ? (
            <Button
              onClick={handleNext}
              disabled={!getStepValidation(currentStep).isValid}
              className="flex items-center gap-2"
            >
              Next
              <ArrowRight className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              onClick={handleSubmit}
              disabled={!getStepValidation(currentStep).isValid || isSubmitting}
              className="flex items-center gap-2"
            >
              {isSubmitting ? 'Submitting...' : 'Submit Assessment'}
              <CheckCircle className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
    <Footer language="en" />
    <Toaster />
    </>
  );

  function renderStep() {
    switch (currentStep) {
      case 1:
        return <BasicInformationStep />;
      case 2:
        return <PhysicalHealthStep />;
      case 3:
        return <MentalWellbeingStep />;
      case 4:
        return <LifestyleFactorsStep />;
      case 5:
        return <ReviewStep />;
      default:
        return null;
    }
  }
}

// Step Components
function BasicInformationStep() {
  const { assessmentData, updateAssessmentData, getStepValidation } = useAssessmentStore();
  const validation = getStepValidation(1);

  return (
    <div className="space-y-6">
      {/* Show validation errors in development */}
      {process.env.NODE_ENV === 'development' && !validation.isValid && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h4 className="text-red-800 font-semibold mb-2">Validation Errors:</h4>
          <ul className="list-disc list-inside text-red-700">
            {validation.errors.map((error, index) => (
              <li key={index}>{error}</li>
            ))}
          </ul>
        </div>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="space-y-2">
          <Label htmlFor="fullName">Full Name *</Label>
          <Input
            id="fullName"
            value={assessmentData.fullName || ''}
            onChange={(e) => updateAssessmentData('fullName', e.target.value)}
            placeholder="Enter your full name"
            required
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="age">Age *</Label>
          <Input
            id="age"
            type="number"
            value={assessmentData.age || ''}
            onChange={(e) => updateAssessmentData('age', parseInt(e.target.value))}
            placeholder="Enter your age"
            min="13"
            max="120"
            required
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="gender">Gender *</Label>
          <Select
            value={assessmentData.gender || ''}
            onValueChange={(value) => updateAssessmentData('gender', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select gender" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="male">Male</SelectItem>
              <SelectItem value="female">Female</SelectItem>
              <SelectItem value="non-binary">Non-binary</SelectItem>
              <SelectItem value="prefer-not-to-say">Prefer not to say</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="occupation">Occupation</Label>
          <Input
            id="occupation"
            value={assessmentData.occupation || ''}
            onChange={(e) => updateAssessmentData('occupation', e.target.value)}
            placeholder="Enter your occupation"
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="email">Email Address *</Label>
        <Input
          id="email"
          type="email"
          value={assessmentData.email || ''}
          onChange={(e) => updateAssessmentData('email', e.target.value)}
          placeholder="Enter your email address"
          required
        />
      </div>
    </div>
  );
}

function PhysicalHealthStep() {
  const { assessmentData, updateAssessmentData } = useAssessmentStore();

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <Label className="text-base font-medium">Sleep Pattern</Label>
          <p className="text-sm text-gray-600 mb-3">How many hours do you typically sleep per night?</p>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500">4h</span>
            <Slider
              value={[assessmentData.sleepHours || 7]}
              onValueChange={(value) => updateAssessmentData('sleepHours', value[0])}
              max={12}
              min={4}
              step={0.5}
              className="flex-1"
            />
            <span className="text-sm text-gray-500">12h</span>
          </div>
          <div className="text-center mt-2">
            <Badge variant="secondary" className="text-lg px-4 py-2">
              {assessmentData.sleepHours || 7} hours
            </Badge>
          </div>
        </div>

        <div>
          <Label className="text-base font-medium">Sleep Quality</Label>
          <p className="text-sm text-gray-600 mb-3">How would you rate your overall sleep quality?</p>
          <Select
            value={assessmentData.sleepQuality || ''}
            onValueChange={(value) => updateAssessmentData('sleepQuality', value)}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select sleep quality..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="poor">Poor - Frequently wake up, hard to fall asleep</SelectItem>
              <SelectItem value="fair">Fair - Some sleep disturbances</SelectItem>
              <SelectItem value="good">Good - Generally restful sleep</SelectItem>
              <SelectItem value="excellent">Excellent - Deep, uninterrupted sleep</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label className="text-base font-medium">Exercise Frequency</Label>
          <p className="text-sm text-gray-600 mb-3">How often do you exercise per week?</p>
          <RadioGroup
            value={assessmentData.exerciseFrequency?.toString() || '3'}
            onValueChange={(value) => updateAssessmentData('exerciseFrequency', parseInt(value))}
          >
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="0" id="exercise-0" />
                <Label htmlFor="exercise-0">Never</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="1" id="exercise-1" />
                <Label htmlFor="exercise-1">1-2 times</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="3" id="exercise-3" />
                <Label htmlFor="exercise-3">3-4 times</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="5" id="exercise-5" />
                <Label htmlFor="exercise-5">5+ times</Label>
              </div>
            </div>
          </RadioGroup>
        </div>

        <div>
          <Label className="text-base font-medium">Diet Quality</Label>
          <p className="text-sm text-gray-600 mb-3">How would you rate your overall diet quality?</p>
          <div className="grid grid-cols-5 gap-2">
            {[1, 2, 3, 4, 5].map((rating) => (
              <Button
                key={rating}
                variant={assessmentData.dietQuality === rating ? 'default' : 'outline'}
                onClick={() => updateAssessmentData('dietQuality', rating)}
                className="h-12"
              >
                {rating}
              </Button>
            ))}
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Poor</span>
            <span>Excellent</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function MentalWellbeingStep() {
  const { assessmentData, updateAssessmentData } = useAssessmentStore();

  const phq9Questions = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed. Or the opposite being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself",
  ];

  const handlePHQ9Change = (questionIndex: number, value: number) => {
    const newPhq9 = { ...assessmentData.phq9, [questionIndex + 1]: value };
    updateAssessmentData('phq9', newPhq9);
  };

  const options = [
    { value: 0, label: "Not at all" },
    { value: 1, label: "Several days" },
    { value: 2, label: "More than half the days" },
    { value: 3, label: "Nearly every day" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Mental Health Screening</h2>
        <p className="text-gray-600 mb-6">Over the last 2 weeks, how often have you been bothered by any of the following problems?</p>
      </div>

      <div className="space-y-8">
        {phq9Questions.map((question, index) => (
          <div key={index} className="bg-gray-50 p-6 rounded-lg">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              {index + 1}. {question}
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {options.map((option) => (
                <button
                  key={option.value}
                  onClick={() => handlePHQ9Change(index, option.value)}
                  className={`p-4 rounded-lg border text-center transition-all ${
                    assessmentData.phq9?.[(index + 1) as keyof typeof assessmentData.phq9] === option.value
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <div className="font-medium">{option.value}</div>
                  <div className="text-sm mt-1">{option.label}</div>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="space-y-4">
        <div>
          <Label className="text-base font-medium">Overall Stress Level</Label>
          <p className="text-sm text-gray-600 mb-3">How would you rate your current stress level?</p>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500">Low</span>
            <Slider
              value={[assessmentData.stressLevel || 5]}
              onValueChange={(value) => updateAssessmentData('stressLevel', value[0])}
              max={10}
              min={1}
              step={1}
              className="flex-1"
            />
            <span className="text-sm text-gray-500">High</span>
          </div>
          <div className="text-center mt-2">
            <Badge 
              variant={assessmentData.stressLevel && assessmentData.stressLevel > 7 ? 'destructive' : 
                      assessmentData.stressLevel && assessmentData.stressLevel > 4 ? 'secondary' : 'default'}
              className="text-lg px-4 py-2"
            >
              {assessmentData.stressLevel || 5}/10
            </Badge>
          </div>
        </div>
      </div>
    </div>
  );
}

function LifestyleFactorsStep() {
  const { assessmentData, updateAssessmentData } = useAssessmentStore();

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <Label className="text-base font-medium">Social Support</Label>
          <p className="text-sm text-gray-600 mb-3">How would you describe your current social support system?</p>
          <Select 
            value={assessmentData.socialSupport || ''} 
            onValueChange={(value) => updateAssessmentData('socialSupport', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select your social support level" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="excellent">Excellent - Strong support from family/friends</SelectItem>
              <SelectItem value="good">Good - Adequate support available</SelectItem>
              <SelectItem value="fair">Fair - Some support but limited</SelectItem>
              <SelectItem value="poor">Poor - Little to no support</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label className="text-base font-medium">Screen Time</Label>
          <p className="text-sm text-gray-600 mb-3">On average, how many hours per day do you spend on screens (TV, computer, phone)?</p>
          <Select 
            value={assessmentData.screenTime || ''} 
            onValueChange={(value) => updateAssessmentData('screenTime', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select your daily screen time" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="less-than-2">Less than 2 hours</SelectItem>
              <SelectItem value="2-4">2-4 hours</SelectItem>
              <SelectItem value="4-6">4-6 hours</SelectItem>
              <SelectItem value="6-8">6-8 hours</SelectItem>
              <SelectItem value="more-than-8">More than 8 hours</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div>
          <Label className="text-base font-medium">Sleep Quality</Label>
          <p className="text-sm text-gray-600 mb-3">How would you rate your overall sleep quality?</p>
          <RadioGroup
            value={assessmentData.sleepQuality || ''}
            onValueChange={(value) => updateAssessmentData('sleepQuality', value)}
          >
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="excellent" id="sleep-excellent" />
                <Label htmlFor="sleep-excellent">Excellent</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="good" id="sleep-good" />
                <Label htmlFor="sleep-good">Good</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="fair" id="sleep-fair" />
                <Label htmlFor="sleep-fair">Fair</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="poor" id="sleep-poor" />
                <Label htmlFor="sleep-poor">Poor</Label>
              </div>
            </div>
          </RadioGroup>
        </div>
      </div>
    </div>
  );
}

function ReviewStep() {
  const { assessmentData } = useAssessmentStore();

  const completionStats = {
    demographics: !!(assessmentData.fullName && assessmentData.age && assessmentData.gender),
    physicalHealth: !!(assessmentData.sleepHours && assessmentData.sleepQuality && assessmentData.exerciseFrequency),
    mentalWellbeing: !!(assessmentData.phq9 && Object.values(assessmentData.phq9).filter(v => v !== undefined).length >= 9),
    lifestyle: !!(assessmentData.socialSupport && assessmentData.screenTime)
  };

  const overallCompletion = Object.values(completionStats).filter(Boolean).length / Object.keys(completionStats).length * 100;

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Review Your Assessment</h2>
        <p className="text-gray-600">Please review your responses before submitting</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Completion Status</CardTitle>
          <CardDescription>Overall completion: {Math.round(overallCompletion)}%</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {Object.entries(completionStats).map(([section, isComplete]) => (
              <div key={section} className="flex items-center justify-between">
                <span className="capitalize">{section.replace(/([A-Z])/g, ' $1').trim()}</span>
                <Badge variant={isComplete ? 'default' : 'secondary'}>
                  {isComplete ? 'Complete' : 'Incomplete'}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Assessment Summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <strong>Name:</strong> {assessmentData.fullName || 'Not provided'}
          </div>
          <div>
            <strong>Age:</strong> {assessmentData.age || 'Not provided'}
          </div>
          <div>
            <strong>Sleep Hours:</strong> {assessmentData.sleepHours || 'Not provided'} hours
          </div>
          <div>
            <strong>Sleep Quality:</strong> {assessmentData.sleepQuality || 'Not provided'}
          </div>
          <div>
            <strong>Exercise Frequency:</strong> {assessmentData.exerciseFrequency || 'Not provided'} times per week
          </div>
          <div>
            <strong>PHQ-9 Responses:</strong> {
              assessmentData.phq9 ? 
                `${Object.values(assessmentData.phq9).filter(v => v !== undefined).length}/9 questions answered` 
                : 'Not completed'
            }
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
