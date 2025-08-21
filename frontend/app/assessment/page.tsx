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

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <div>
          <Label className="text-base font-medium">Stress Level</Label>
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

        <div>
          <Label className="text-base font-medium">Mood Assessment</Label>
          <p className="text-sm text-gray-600 mb-3">How would you describe your overall mood recently?</p>
          <RadioGroup
            value={assessmentData.moodScore?.toString() || '6'}
            onValueChange={(value) => updateAssessmentData('moodScore', parseInt(value))}
          >
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="1" id="mood-1" />
                <Label htmlFor="mood-1">Very Low (1-2)</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="3" id="mood-3" />
                <Label htmlFor="mood-3">Low (3-4)</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="6" id="mood-6" />
                <Label htmlFor="mood-6">Moderate (5-7)</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="9" id="mood-9" />
                <Label htmlFor="mood-9">High (8-10)</Label>
              </div>
            </div>
          </RadioGroup>
        </div>

        <div>
          <Label className="text-base font-medium">Mental Health History</Label>
          <p className="text-sm text-gray-600 mb-3">Have you experienced any mental health challenges?</p>
          <div className="space-y-3">
            {[
              'Anxiety',
              'Depression',
              'Stress-related issues',
              'Sleep disorders',
              'None of the above'
            ].map((condition) => (
              <div key={condition} className="flex items-center space-x-2">
                <Checkbox
                  id={condition}
                  checked={assessmentData.mentalHealthHistory?.includes(condition) || false}
                  onCheckedChange={(checked) => {
                    const current = assessmentData.mentalHealthHistory || [];
                    if (checked) {
                      updateAssessmentData('mentalHealthHistory', [...current, condition]);
                    } else {
                      updateAssessmentData('mentalHealthHistory', current.filter(c => c !== condition));
                    }
                  }}
                />
                <Label htmlFor={condition} className="text-sm">{condition}</Label>
              </div>
            ))}
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
          <Label className="text-base font-medium">Social Connections</Label>
          <p className="text-sm text-gray-600 mb-3">How would you rate your social connections and support network?</p>
          <div className="grid grid-cols-5 gap-2">
            {[1, 2, 3, 4, 5].map((rating) => (
              <Button
                key={rating}
                variant={assessmentData.socialConnections === rating ? 'default' : 'outline'}
                onClick={() => updateAssessmentData('socialConnections', rating)}
                className="h-12"
              >
                {rating}
              </Button>
            ))}
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Very Low</span>
            <span>Very High</span>
          </div>
        </div>

        <div>
          <Label className="text-base font-medium">Work-Life Balance</Label>
          <p className="text-sm text-gray-600 mb-3">How would you rate your work-life balance?</p>
          <RadioGroup
            value={assessmentData.workLifeBalance?.toString() || '3'}
            onValueChange={(value) => updateAssessmentData('workLifeBalance', parseInt(value))}
          >
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="1" id="balance-1" />
                <Label htmlFor="balance-1">Poor (1-2)</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="3" id="balance-3" />
                <Label htmlFor="balance-3">Fair (3-4)</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="6" id="balance-6" />
                <Label htmlFor="balance-6">Good (5-7)</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="9" id="balance-9" />
                <Label htmlFor="balance-9">Excellent (8-10)</Label>
              </div>
            </div>
          </RadioGroup>
        </div>

        <div>
          <Label className="text-base font-medium">Financial Stress</Label>
          <p className="text-sm text-gray-600 mb-3">How would you rate your current financial stress level?</p>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500">None</span>
            <Slider
              value={[assessmentData.financialStress || 5]}
              onValueChange={(value) => updateAssessmentData('financialStress', value[0])}
              max={10}
              min={1}
              step={1}
              className="flex-1"
            />
            <span className="text-sm text-gray-500">High</span>
          </div>
          <div className="text-center mt-2">
            <Badge 
              variant={assessmentData.financialStress && assessmentData.financialStress > 7 ? 'destructive' : 
                      assessmentData.financialStress && assessmentData.financialStress > 4 ? 'secondary' : 'default'}
              className="text-lg px-4 py-2"
            >
              {assessmentData.financialStress || 5}/10
            </Badge>
          </div>
        </div>

        <div>
          <Label htmlFor="additionalNotes">Additional Notes</Label>
          <Textarea
            id="additionalNotes"
            value={assessmentData.additionalNotes || ''}
            onChange={(e) => updateAssessmentData('additionalNotes', e.target.value)}
            placeholder="Any additional information you'd like to share..."
            rows={3}
          />
        </div>
      </div>
    </div>
  );
}

function ReviewStep() {
  const { assessmentData } = useAssessmentStore();

  const renderField = (label: string, value: any, type: 'text' | 'rating' | 'list' = 'text') => {
    if (value === undefined || value === null || value === '') return null;

    return (
      <div className="flex justify-between py-2 border-b border-gray-100">
        <span className="font-medium text-gray-700">{label}</span>
        <span className="text-gray-900">
          {type === 'rating' && typeof value === 'number' ? `${value}/10` :
           type === 'list' && Array.isArray(value) ? value.join(', ') :
           value}
        </span>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 p-4 rounded-lg">
        <h3 className="font-medium text-blue-900 mb-2">Review Your Assessment</h3>
        <p className="text-sm text-blue-700">
          Please review your responses before submitting. You can go back to any step to make changes.
        </p>
      </div>

      <div className="space-y-4">
        <h4 className="font-medium text-gray-900">Basic Information</h4>
        <div className="bg-gray-50 p-4 rounded-lg space-y-1">
          {renderField('Full Name', assessmentData.fullName)}
          {renderField('Age', assessmentData.age)}
          {renderField('Gender', assessmentData.gender)}
          {renderField('Occupation', assessmentData.occupation)}
          {renderField('Email', assessmentData.email)}
        </div>

        <h4 className="font-medium text-gray-900">Physical Health</h4>
        <div className="bg-gray-50 p-4 rounded-lg space-y-1">
          {renderField('Sleep Hours', assessmentData.sleepHours)}
          {renderField('Exercise Frequency', assessmentData.exerciseFrequency, 'text')}
          {renderField('Diet Quality', assessmentData.dietQuality, 'rating')}
        </div>

        <h4 className="font-medium text-gray-900">Mental Wellbeing</h4>
        <div className="bg-gray-50 p-4 rounded-lg space-y-1">
          {renderField('Stress Level', assessmentData.stressLevel, 'rating')}
          {renderField('Mood Score', assessmentData.moodScore, 'rating')}
          {renderField('Mental Health History', assessmentData.mentalHealthHistory, 'list')}
        </div>

        <h4 className="font-medium text-gray-900">Lifestyle Factors</h4>
        <div className="bg-gray-50 p-4 rounded-lg space-y-1">
          {renderField('Social Connections', assessmentData.socialConnections, 'rating')}
          {renderField('Work-Life Balance', assessmentData.workLifeBalance, 'rating')}
          {renderField('Financial Stress', assessmentData.financialStress, 'rating')}
          {renderField('Additional Notes', assessmentData.additionalNotes)}
        </div>
      </div>

      <div className="bg-green-50 p-4 rounded-lg">
        <h3 className="font-medium text-green-900 mb-2">Ready to Submit</h3>
        <p className="text-sm text-green-700">
          Your assessment is complete! Click submit to receive your personalized health insights and recommendations.
        </p>
      </div>
    </div>
  );
}
