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
  riskLevel: string
  phq9Score: number
  confidenceScore: number
  emergencyResources: {
    name: string
    phone: string
    description: string
    available24h: boolean
  }[]
  riskFactors: string[]
  protectiveFactors: string[]
  recommendations: string[]
  culturalConsiderations: string[]
}


export interface AssessmentHistory {
  date: string
  phq9Score: number
  riskLevel: "low" | "moderate" | "high"
  sleepHours?: string
  sleepQuality?: string
  sleepHoursNumeric?: number
  stressLevel?: string
  exerciseFrequency?: string
  socialSupport?: string
}

export interface HistoricalTrend {
  overallTrend: "improving" | "stable" | "declining"
  phq9Trend: number
  sleepTrend: "improving" | "stable" | "declining"
  insights: string[]
  recommendations: string[]
  correlations: Record<string, number>
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
