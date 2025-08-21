import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface AssessmentData {
  // Basic Information
  fullName?: string;
  age?: number;
  gender?: string;
  occupation?: string;
  email?: string;
  
  // PHQ-9 Data (proper format)
  phq9?: {
    1?: number;
    2?: number;
    3?: number;
    4?: number;
    5?: number;
    6?: number;
    7?: number;
    8?: number;
    9?: number;
  };
  
  // Physical Health
  sleepHours?: number;
  sleepQuality?: string;
  exerciseFrequency?: number;
  
  // Sleep and lifestyle data
  stressLevel?: number;
  socialSupport?: string;
  screenTime?: string;
  
  // Metadata
  completedAt?: string;
  lastModified?: string;
}

// New interface for the transformed data that matches backend expectations
export interface TransformedAssessmentData {
  demographics: {
    age: string;
    gender: string;
    region: string;
    education: string;
    employmentStatus: string;
  };
  phq9: {
    [key: string]: number | null;
  };
  sleep: {
    sleepHours: string;
    sleepQuality: string;
    exerciseFrequency: string;
    stressLevel: string;
    socialSupport: string;
    screenTime: string;
  };
  language: string;
}

interface AssessmentStore {
  // State
  assessmentData: AssessmentData;
  currentStep: number;
  isCompleted: boolean;
  
  // Actions
  updateAssessmentData: (field: keyof AssessmentData, value: any) => void;
  setCurrentStep: (step: number) => void;
  resetAssessment: () => void;
  completeAssessment: () => void;
  
  // Validation
  validateStep: (step: number) => boolean;
  getStepValidation: (step: number) => { isValid: boolean; errors: string[] };
  
  // Data transformation
  getTransformedData: () => TransformedAssessmentData;
  
  // Utilities
  getProgress: () => number;
  canProceedToStep: (step: number) => boolean;
}

const initialAssessmentData: AssessmentData = {
  fullName: '',
  age: undefined,
  gender: '',
  occupation: '',
  email: '',
  phq9: {
    1: undefined,
    2: undefined,
    3: undefined,
    4: undefined,
    5: undefined,
    6: undefined,
    7: undefined,
    8: undefined,
    9: undefined,
  },
  sleepHours: 7,
  sleepQuality: '',
  exerciseFrequency: 3,
  stressLevel: 5,
  socialSupport: '',
  screenTime: '',
  completedAt: undefined,
  lastModified: new Date().toISOString()
};

export const useAssessmentStore = create<AssessmentStore>()(
  persist(
    (set, get) => ({
      // Initial state
      assessmentData: initialAssessmentData,
      currentStep: 1,
      isCompleted: false,

      // Actions
      updateAssessmentData: (field: keyof AssessmentData, value: any) => {
        set((state) => ({
          assessmentData: {
            ...state.assessmentData,
            [field]: value,
            lastModified: new Date().toISOString()
          }
        }));
      },

      setCurrentStep: (step: number) => {
        set({ currentStep: step });
      },

      resetAssessment: () => {
        set({
          assessmentData: initialAssessmentData,
          currentStep: 1,
          isCompleted: false
        });
      },

      completeAssessment: () => {
        set((state) => ({
          assessmentData: {
            ...state.assessmentData,
            completedAt: new Date().toISOString()
          },
          isCompleted: true
        }));
      },

      // Validation
      validateStep: (step: number) => {
        const validation = get().getStepValidation(step);
        return validation.isValid;
      },

      getStepValidation: (step: number) => {
        const { assessmentData } = get();
        const errors: string[] = [];

        switch (step) {
          case 1: // Basic Information
            if (!assessmentData.fullName?.trim()) {
              errors.push('Full name is required');
            }
            if (!assessmentData.age || assessmentData.age < 13 || assessmentData.age > 120) {
              errors.push('Valid age (13-120) is required');
            }
            if (!assessmentData.gender?.trim()) {
              errors.push('Gender is required');
            }
            break;

          case 2: // Physical Health
            if (assessmentData.sleepHours === undefined || assessmentData.sleepHours < 4 || assessmentData.sleepHours > 12) {
              errors.push('Sleep hours must be between 4 and 12');
            }
            if (!assessmentData.sleepQuality?.trim()) {
              errors.push('Sleep quality is required');
            }
            if (assessmentData.exerciseFrequency === undefined || assessmentData.exerciseFrequency < 0 || assessmentData.exerciseFrequency > 7) {
              errors.push('Exercise frequency must be between 0 and 7');
            }
            break;

          case 3: // Mental Wellbeing (PHQ-9)
            if (!assessmentData.phq9) {
              errors.push('PHQ-9 responses are required');
            } else {
              const answeredQuestions = Object.values(assessmentData.phq9).filter(v => v !== undefined && v !== null);
              if (answeredQuestions.length < 9) {
                errors.push('All PHQ-9 questions must be answered');
              }
            }
            if (assessmentData.stressLevel === undefined || assessmentData.stressLevel < 1 || assessmentData.stressLevel > 10) {
              errors.push('Stress level must be between 1 and 10');
            }
            break;

          case 4: // Lifestyle Factors
            if (!assessmentData.socialSupport?.trim()) {
              errors.push('Social support information is required');
            }
            if (!assessmentData.screenTime?.trim()) {
              errors.push('Screen time information is required');
            }
            break;

          case 5: // Review & Submit
            // All previous steps must be valid
            for (let i = 1; i <= 4; i++) {
              if (!get().validateStep(i)) {
                errors.push(`Step ${i} is not complete`);
              }
            }
            break;

          default:
            errors.push('Invalid step number');
        }

        return {
          isValid: errors.length === 0,
          errors
        };
      },

      // Data transformation
      getTransformedData: () => {
        const { assessmentData } = get();
        return {
          demographics: {
            age: assessmentData.age?.toString() || '',
            gender: assessmentData.gender || '',
            region: '', // Placeholder, will be populated from backend
            education: '', // Placeholder, will be populated from backend
            employmentStatus: '', // Placeholder, will be populated from backend
          },
          phq9: assessmentData.phq9 || {},
          sleep: {
            sleepHours: assessmentData.sleepHours?.toString() || '',
            sleepQuality: assessmentData.sleepQuality || '',
            exerciseFrequency: assessmentData.exerciseFrequency?.toString() || '',
            stressLevel: assessmentData.stressLevel?.toString() || '',
            socialSupport: assessmentData.socialSupport || '',
            screenTime: assessmentData.screenTime || '',
          },
          language: 'en',
        };
      },

      // Utilities
      getProgress: () => {
        const { currentStep } = get();
        return (currentStep / 5) * 100;
      },

      canProceedToStep: (step: number) => {
        // Can only proceed to next step if current step is valid
        const { currentStep } = get();
        if (step <= currentStep) return true;
        
        // Check if all previous steps are valid
        for (let i = 1; i < step; i++) {
          if (!get().validateStep(i)) {
            return false;
          }
        }
        return true;
      }
    }),
    {
      name: 'mindguard-assessment-storage',
      partialize: (state) => ({
        assessmentData: state.assessmentData,
        currentStep: state.currentStep,
        isCompleted: state.isCompleted
      })
    }
  )
);

// Helper function to validate email
function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// Export additional utilities
export const getAssessmentSummary = (data: AssessmentData) => {
  const summary = {
    totalFields: 0,
    completedFields: 0,
    completionPercentage: 0,
    riskIndicators: [] as string[]
  };

  // Count total and completed fields
  Object.entries(data).forEach(([key, value]) => {
    if (key !== 'completedAt' && key !== 'lastModified' && key !== 'phq9') {
      summary.totalFields++;
      if (value !== undefined && value !== null && value !== '') {
        if (Array.isArray(value) && value.length > 0) {
          summary.completedFields++;
        } else if (!Array.isArray(value)) {
          summary.completedFields++;
        }
      }
    }
  });

  // Count PHQ-9 separately
  if (data.phq9) {
    summary.totalFields += 9; // 9 PHQ-9 questions
    const answeredQuestions = Object.values(data.phq9).filter(v => v !== undefined && v !== null);
    summary.completedFields += answeredQuestions.length;
  }

  summary.completionPercentage = Math.round((summary.completedFields / summary.totalFields) * 100);

  // Identify potential risk indicators
  if (data.sleepHours && data.sleepHours < 6) {
    summary.riskIndicators.push('Insufficient sleep');
  }
  if (data.stressLevel && data.stressLevel > 7) {
    summary.riskIndicators.push('High stress levels');
  }
  if (data.exerciseFrequency && data.exerciseFrequency < 2) {
    summary.riskIndicators.push('Low exercise frequency');
  }
  if (data.phq9) {
    const phq9Score = Object.values(data.phq9).reduce((sum, val) => sum + (val || 0), 0);
    if (phq9Score > 14) {
      summary.riskIndicators.push('Elevated depression screening score');
    }
  }

  return summary;
};

export const calculateHealthScore = (data: AssessmentData): number => {
  let score = 0;
  let maxScore = 0;

  // Sleep score (0-25 points)
  if (data.sleepHours) {
    maxScore += 25;
    if (data.sleepHours >= 7 && data.sleepHours <= 9) {
      score += 25; // Optimal sleep
    } else if (data.sleepHours >= 6 && data.sleepHours <= 10) {
      score += 20; // Good sleep
    } else if (data.sleepHours >= 5 && data.sleepHours <= 11) {
      score += 15; // Acceptable sleep
    } else {
      score += 5; // Poor sleep
    }
  }

  // Exercise score (0-20 points)
  if (data.exerciseFrequency !== undefined) {
    maxScore += 20;
    if (data.exerciseFrequency >= 5) {
      score += 20; // Excellent
    } else if (data.exerciseFrequency >= 3) {
      score += 15; // Good
    } else if (data.exerciseFrequency >= 1) {
      score += 10; // Fair
    } else {
      score += 5; // Poor
    }
  }

  // Stress score (0-20 points) - inverse relationship
  if (data.stressLevel) {
    maxScore += 20;
    if (data.stressLevel <= 3) {
      score += 20; // Low stress
    } else if (data.stressLevel <= 5) {
      score += 15; // Moderate stress
    } else if (data.stressLevel <= 7) {
      score += 10; // High stress
    } else {
      score += 5; // Very high stress
    }
  }

  // PHQ-9 score (0-35 points) - inverse relationship
  if (data.phq9) {
    maxScore += 35;
    const phq9Score = Object.values(data.phq9).reduce((sum, val) => sum + (val || 0), 0);
    if (phq9Score <= 4) {
      score += 35; // Minimal depression
    } else if (phq9Score <= 9) {
      score += 25; // Mild depression
    } else if (phq9Score <= 14) {
      score += 15; // Moderate depression
    } else if (phq9Score <= 19) {
      score += 10; // Moderately severe depression
    } else {
      score += 5; // Severe depression
    }
  }

  return maxScore > 0 ? Math.round((score / maxScore) * 100) : 0;
};

export const getRiskLevel = (healthScore: number): 'low' | 'normal' | 'high' => {
  if (healthScore >= 80) return 'low';
  if (healthScore >= 60) return 'normal';
  return 'high';
}; 