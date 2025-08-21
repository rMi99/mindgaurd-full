'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
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
import { ArrowLeft, ArrowRight, CheckCircle, Brain, Heart, Activity, Users, Moon, Utensils } from 'lucide-react';
import { useAssessmentStore } from '@/lib/stores/assessmentStore';
import { useAuthStore } from '@/lib/stores/authStore';

interface AssessmentStep {
  id: number;
  title: string;
  description: string;
  icon: React.ReactNode;
  isCompleted: boolean;
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
  
  const { isAuthenticated, user } = useAuthStore();
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
      if (!token) {
        // Store the intended destination and redirect to login
        localStorage.setItem('redirect_after_login', '/assessment');
        router.push('/auth');
        return;
      }
      setIsLoading(false);
    };

    checkAuth();
  }, [router]);

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

  // If not authenticated, don't render the assessment
  if (!isAuthenticated) {
    return null;
  }

  const totalSteps = ASSESSMENT_STEPS.length;
  const progress = (currentStep / totalSteps) * 100;

  const handleNext = () => {
    if (validateStep(currentStep)) {
      if (currentStep < totalSteps) {
        setCurrentStep(currentStep + 1);
      }
    }
  };

  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async () => {
    if (!validateStep(currentStep)) return;
    
    setIsSubmitting(true);
    try {
      const token = localStorage.getItem('mindguard_token') || localStorage.getItem('access_token');
      
      // Use the transformed data from the store
      const transformedData = getTransformedData();

      // Submit assessment data to API
      const response = await fetch('/api/assessment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(transformedData),
      });

      if (response.ok) {
        const result = await response.json();
        // Store the comprehensive result for the results page
        localStorage.setItem('assessment_result', JSON.stringify(result));
        // Redirect to results page
        router.push(`/assessment/results?assessment_id=${result.assessment_id}`);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to submit assessment');
      }
    } catch (error) {
      console.error('Error submitting assessment:', error);
      // Handle error (show toast, etc.)
      alert('Failed to submit assessment. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderStepContent = () => {
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
  };

  const canProceed = () => {
    return validateStep(currentStep);
  };

  const canGoBack = () => {
    return currentStep > 1;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Health Assessment
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Complete this comprehensive assessment to receive personalized health insights and recommendations.
            Your responses help us understand your current health status and provide tailored guidance.
          </p>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">
              Step {currentStep} of {totalSteps}
            </span>
            <span className="text-sm font-medium text-gray-700">
              {Math.round(progress)}% Complete
            </span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        {/* Step Navigation */}
        <div className="flex justify-between items-center mb-8 overflow-x-auto">
          {ASSESSMENT_STEPS.map((step) => (
            <div
              key={step.id}
              className={`flex flex-col items-center min-w-0 flex-1 ${
                step.id === currentStep ? 'text-blue-600' : 
                step.id < currentStep ? 'text-green-600' : 'text-gray-400'
              }`}
            >
              <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-2 ${
                step.id === currentStep ? 'bg-blue-600 text-white' :
                step.id < currentStep ? 'bg-green-600 text-white' : 'bg-gray-200'
              }`}>
                {step.id < currentStep ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  step.icon
                )}
              </div>
              <div className="text-center">
                <div className="text-xs font-medium truncate">{step.title}</div>
                <div className="text-xs text-gray-500 hidden sm:block">{step.description}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Step Content */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              {ASSESSMENT_STEPS[currentStep - 1].icon}
              {ASSESSMENT_STEPS[currentStep - 1].title}
            </CardTitle>
            <p className="text-gray-600">
              {ASSESSMENT_STEPS[currentStep - 1].description}
            </p>
          </CardHeader>
          <CardContent>
            {renderStepContent()}
          </CardContent>
        </Card>

        {/* Navigation Buttons */}
        <div className="flex justify-between">
          <Button
            variant="outline"
            onClick={handlePrevious}
            disabled={!canGoBack()}
            className="flex items-center gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Previous
          </Button>

          <div className="flex gap-3">
            {currentStep < totalSteps ? (
              <Button
                onClick={handleNext}
                disabled={!canProceed()}
                className="flex items-center gap-2"
              >
                Next
                <ArrowRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={handleSubmit}
                disabled={!canProceed() || isSubmitting}
                className="flex items-center gap-2 bg-green-600 hover:bg-green-700"
              >
                {isSubmitting ? 'Submitting...' : 'Submit Assessment'}
                <CheckCircle className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {/* Save Progress */}
        <div className="text-center mt-6">
          <Button
            variant="ghost"
            onClick={() => {
              // Save progress to localStorage
              localStorage.setItem('assessmentProgress', JSON.stringify({
                currentStep,
                data: assessmentData
              }));
            }}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            Save Progress
          </Button>
        </div>
      </div>
    </div>
  );
}

// Step Components
function BasicInformationStep() {
  const { assessmentData, updateAssessmentData } = useAssessmentStore();

  return (
    <div className="space-y-6">
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
          <Label htmlFor="gender">Gender</Label>
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
                    assessmentData.phq9?.[index + 1] === option.value
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
