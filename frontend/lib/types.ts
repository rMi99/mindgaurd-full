export interface DemographicData {
  age: string
  gender: string
  region: string
  education: string
  employmentStatus: string
}

export interface PHQ9Data {
  [key: string]: number | null  // Frontend sends as {"1": 0, "2": 1, etc.}
}

export interface SleepData {
  sleepHours: string
  sleepQuality: string
  exerciseFrequency: string
  stressLevel: string
  socialSupport: string
  screenTime: string
}

export interface AssessmentData {
  demographics: DemographicData
  phq9: PHQ9Data
  sleep: SleepData
  language: string
}

// export interface AssessmentResult {
//   phq9Score: number
//   riskLevel: "low" | "moderate" | "high"
//   scores: number[]
//   riskFactors: string[]
//   recommendations: string[]
// }

// export interface AssessmentResult {
//   phq9Score: number
//   riskLevel: "low" | "moderate" | "high"
//   scores: number[]
//   riskFactors: string[]
//   recommendations: string[]
//   emergencyResources: EmergencyResource[]  // Add this property to include emergency resources
//   culturalConsiderations: string[]  // Add this property to include cultural considerations
//   confidenceScore: number  // Add this property for confidence score
// }

export interface AssessmentResult {
  assessment_id: string
  riskLevel: 'low' | 'moderate' | 'high'
  phq9Score: number
  confidenceScore: number
  riskFactors: string[]
  protectiveFactors: string[]
  recommendations: string[]
  culturalConsiderations: string[]
  emergencyResources: Array<{
    name: string
    phone: string
    description: string
    available24h: boolean
  }>
  brainHealActivities: Array<{
    name: string
    duration: string
    description: string
    steps: string[]
    benefits: string[]
    difficulty: string
  }>
  weeklyPlan: Record<string, string[]>
  created_at: string
}

export interface AssessmentHistory {
  id: string
  date: string
  phq9Score: number
  riskLevel: string
  sleepHours?: string
  sleepQuality?: string
  sleepHoursNumeric?: number
  stressLevel?: string
  exerciseFrequency?: string
  socialSupport?: string
}

export interface HistoricalTrend {
  overallTrend: string
  phq9Trend: number
  sleepTrend: string
  insights: string[]
  recommendations: string[]
  correlations: Record<string, number>
}

export interface PersonalizedInsights {
  encouragingMessage: string
  psychologicalInsights: string[]
  personalizedRecommendations: string[]
  progressSummary: string
  nextSteps: string[]
}

export interface DashboardData {
  history: AssessmentHistory[]
  trends?: HistoricalTrend
  personalizedInsights?: PersonalizedInsights
  userInfo: Record<string, any>
}

export interface AssessmentData {
  demographics: {
    age: string
    gender: string
    region: string
    education: string
    employmentStatus: string
  }
  phq9: {
    scores: number[]
  }
  sleep: {
    averageHours: number
    weeklyPattern: number[]
  }
  behavioral: {
    moodDescription: string
    stressLevel: number
    socialConnections: number
    physicalActivity: number
  }
}

export interface PyTorchResults {
  risk_level: 'low' | 'moderate' | 'high'
  confidence: number
  key_factors: Record<string, { value: string; impact: string }>
  interventions: {
    immediate: Array<{
      type: string
      duration: string
      reason: string
      title: string
      description: string
    }>
    longterm: Array<{
      type: string
      plan: string
      title: string
      description: string
    }>
  }
  biometric_scores: {
    sleep: number
    mood: number
    social: number
    stress: number
    energy: number
  }
  recommendations: string[]
}

export type Language = 'en' | 'es' | 'fr'

export interface TranslationKeys {
  welcome: string
  assessmentComplete: string
  riskLevel: string
  startNewAssessment: string
  viewDetailedReport: string
  // Add other translation keys as needed
}

// export interface EmergencyResource {
//   name: string
//   phone: string
//   description: string
//   available24h: boolean
// }
export interface EmergencyResource {
  name: string
  phone: string
  description: string
  available24h: boolean
}



export type Language = "en" | "si" | "ta" | "es" | "fr" | "zh"
