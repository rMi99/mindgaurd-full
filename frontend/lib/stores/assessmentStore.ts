import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface AssessmentData {
  // Basic Information
  fullName?: string;
  age?: number;
  gender?: string;
  occupation?: string;
  email?: string;
  
  // Physical Health
  sleepHours?: number;
  exerciseFrequency?: number;
  dietQuality?: number;
  
  // Mental Wellbeing
  stressLevel?: number;
  moodScore?: number;
  mentalHealthHistory?: string[];
  
  // Lifestyle Factors
  socialConnections?: number;
  workLifeBalance?: number;
  financialStress?: number;
  additionalNotes?: string;
  
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
    scores: Record<string, number>;
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
  sleepHours: 7,
  exerciseFrequency: 3,
  dietQuality: 3,
  stressLevel: 5,
  moodScore: 6,
  mentalHealthHistory: [],
  socialConnections: 3,
  workLifeBalance: 3,
  financialStress: 5,
  additionalNotes: '',
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
            if (!assessmentData.email?.trim() || !isValidEmail(assessmentData.email)) {
              errors.push('Valid email address is required');
            }
            break;

          case 2: // Physical Health
            if (assessmentData.sleepHours === undefined || assessmentData.sleepHours < 4 || assessmentData.sleepHours > 12) {
              errors.push('Sleep hours must be between 4 and 12');
            }
            if (assessmentData.exerciseFrequency === undefined || assessmentData.exerciseFrequency < 0 || assessmentData.exerciseFrequency > 7) {
              errors.push('Exercise frequency must be between 0 and 7');
            }
            if (assessmentData.dietQuality === undefined || assessmentData.dietQuality < 1 || assessmentData.dietQuality > 5) {
              errors.push('Diet quality must be between 1 and 5');
            }
            break;

          case 3: // Mental Wellbeing
            if (assessmentData.stressLevel === undefined || assessmentData.stressLevel < 1 || assessmentData.stressLevel > 10) {
              errors.push('Stress level must be between 1 and 10');
            }
            if (assessmentData.moodScore === undefined || assessmentData.moodScore < 1 || assessmentData.moodScore > 10) {
              errors.push('Mood score must be between 1 and 10');
            }
            break;

          case 4: // Lifestyle Factors
            if (assessmentData.socialConnections === undefined || assessmentData.socialConnections < 1 || assessmentData.socialConnections > 5) {
              errors.push('Social connections must be between 1 and 5');
            }
            if (assessmentData.workLifeBalance === undefined || assessmentData.workLifeBalance < 1 || assessmentData.workLifeBalance > 10) {
              errors.push('Work-life balance must be between 1 and 10');
            }
            if (assessmentData.financialStress === undefined || assessmentData.financialStress < 1 || assessmentData.financialStress > 10) {
              errors.push('Financial stress must be between 1 and 10');
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
          phq9: {
            scores: {
              'PHQ-9 Score': assessmentData.stressLevel || 0, // Using stressLevel as a placeholder for PHQ-9 score
              'Mood Score': assessmentData.moodScore || 0, // Using moodScore as a placeholder for PHQ-9 score
            },
          },
          sleep: {
            sleepHours: assessmentData.sleepHours?.toString() || '',
            sleepQuality: '', // Placeholder
            exerciseFrequency: assessmentData.exerciseFrequency?.toString() || '',
            stressLevel: assessmentData.stressLevel?.toString() || '',
            socialSupport: '', // Placeholder
            screenTime: '', // Placeholder
          },
          language: 'en', // Placeholder, will be populated from backend
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
    if (key !== 'completedAt' && key !== 'lastModified') {
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

  summary.completionPercentage = Math.round((summary.completedFields / summary.totalFields) * 100);

  // Identify potential risk indicators
  if (data.sleepHours && data.sleepHours < 6) {
    summary.riskIndicators.push('Insufficient sleep');
  }
  if (data.stressLevel && data.stressLevel > 7) {
    summary.riskIndicators.push('High stress levels');
  }
  if (data.moodScore && data.moodScore < 4) {
    summary.riskIndicators.push('Low mood score');
  }
  if (data.exerciseFrequency && data.exerciseFrequency < 2) {
    summary.riskIndicators.push('Low exercise frequency');
  }
  if (data.socialConnections && data.socialConnections < 3) {
    summary.riskIndicators.push('Limited social connections');
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

  // Mood score (0-20 points)
  if (data.moodScore) {
    maxScore += 20;
    if (data.moodScore >= 8) {
      score += 20; // Excellent mood
    } else if (data.moodScore >= 6) {
      score += 15; // Good mood
    } else if (data.moodScore >= 4) {
      score += 10; // Fair mood
    } else {
      score += 5; // Poor mood
    }
  }

  // Social connections score (0-15 points)
  if (data.socialConnections) {
    maxScore += 15;
    if (data.socialConnections >= 4) {
      score += 15; // Strong connections
    } else if (data.socialConnections >= 3) {
      score += 10; // Moderate connections
    } else {
      score += 5; // Limited connections
    }
  }

  return maxScore > 0 ? Math.round((score / maxScore) * 100) : 0;
};

export const getRiskLevel = (healthScore: number): 'low' | 'normal' | 'high' => {
  if (healthScore >= 80) return 'low';
  if (healthScore >= 60) return 'normal';
  return 'high';
}; 