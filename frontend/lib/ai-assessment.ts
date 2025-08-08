import type { AssessmentData, AssessmentResult, EmergencyResource } from "./types"

export class AIAssessment {
  private calculatePHQ9Score(phq9Data: any): number {
    let score = 0
    Object.values(phq9Data).forEach((value) => {
      if (typeof value === "number") {
        score += value
      }
    })
    return score
  }

  private getRiskLevel(phq9Score: number): "low" | "moderate" | "high" {
    if (phq9Score <= 4) return "low"
    if (phq9Score <= 14) return "moderate"
    return "high"
  }

  private analyzeRiskFactors(data: AssessmentData): string[] {
    const riskFactors: string[] = []

    // PHQ-9 specific risk factors
    if (data.phq9[9] && data.phq9[9] > 0) {
      riskFactors.push("Thoughts of self-harm reported")
    }

    // Sleep-related risk factors
    if (data.sleep.sleepHours === "<4" || data.sleep.sleepHours === "4-5") {
      riskFactors.push("Insufficient sleep (less than 6 hours)")
    }

    if (data.sleep.sleepQuality === "very-poor" || data.sleep.sleepQuality === "poor") {
      riskFactors.push("Poor sleep quality")
    }

    // Lifestyle risk factors
    if (data.sleep.exerciseFrequency === "never") {
      riskFactors.push("Sedentary lifestyle (no exercise)")
    }

    if (data.sleep.stressLevel === "high" || data.sleep.stressLevel === "very-high") {
      riskFactors.push("High stress levels")
    }

    if (data.sleep.socialSupport === "none" || data.sleep.socialSupport === "minimal") {
      riskFactors.push("Limited social support")
    }

    if (data.sleep.screenTime === "8-10" || data.sleep.screenTime === ">10") {
      riskFactors.push("Excessive screen time")
    }

    // Demographic risk factors
    if (data.demographics.employmentStatus === "unemployed") {
      riskFactors.push("Unemployment stress")
    }

    return riskFactors
  }

  private analyzeProtectiveFactors(data: AssessmentData): string[] {
    const protectiveFactors: string[] = []

    // Sleep protective factors
    if (data.sleep.sleepHours === "7-8" || data.sleep.sleepHours === "8-9") {
      protectiveFactors.push("Adequate sleep duration")
    }

    if (data.sleep.sleepQuality === "good" || data.sleep.sleepQuality === "excellent") {
      protectiveFactors.push("Good sleep quality")
    }

    // Lifestyle protective factors
    if (
      data.sleep.exerciseFrequency === "3-4" ||
      data.sleep.exerciseFrequency === "5-6" ||
      data.sleep.exerciseFrequency === "daily"
    ) {
      protectiveFactors.push("Regular physical activity")
    }

    if (data.sleep.stressLevel === "low" || data.sleep.stressLevel === "very-low") {
      protectiveFactors.push("Low stress levels")
    }

    if (data.sleep.socialSupport === "strong" || data.sleep.socialSupport === "excellent") {
      protectiveFactors.push("Strong social support network")
    }

    // Employment stability
    if (
      data.demographics.employmentStatus === "employed-full" ||
      data.demographics.employmentStatus === "employed-part"
    ) {
      protectiveFactors.push("Employment stability")
    }

    // Education
    if (
      data.demographics.education === "bachelor" ||
      data.demographics.education === "master" ||
      data.demographics.education === "doctorate"
    ) {
      protectiveFactors.push("Higher education attainment")
    }

    return protectiveFactors
  }

  private generateRecommendations(data: AssessmentData, riskLevel: string): string[] {
    const recommendations: string[] = []

    // Universal recommendations
    recommendations.push("Consider speaking with a healthcare professional about your mental health")
    recommendations.push("Practice regular self-care activities that bring you joy")

    // Risk-specific recommendations
    if (riskLevel === "high") {
      recommendations.push("Seek immediate professional mental health support")
      recommendations.push("Consider contacting a crisis helpline if you feel overwhelmed")
    }

    if (riskLevel === "moderate") {
      recommendations.push("Schedule an appointment with your primary care doctor")
      recommendations.push("Consider counseling or therapy services")
    }

    // Sleep recommendations
    if (data.sleep.sleepHours === "<4" || data.sleep.sleepHours === "4-5") {
      recommendations.push("Prioritize getting 7-9 hours of sleep per night")
      recommendations.push("Establish a consistent sleep schedule")
    }

    if (data.sleep.sleepQuality === "poor" || data.sleep.sleepQuality === "very-poor") {
      recommendations.push("Create a relaxing bedtime routine")
      recommendations.push("Limit screen time before bed")
    }

    // Exercise recommendations
    if (data.sleep.exerciseFrequency === "never" || data.sleep.exerciseFrequency === "1-2") {
      recommendations.push("Incorporate regular physical activity into your routine")
      recommendations.push("Start with 30 minutes of walking 3-4 times per week")
    }

    // Stress management
    if (data.sleep.stressLevel === "high" || data.sleep.stressLevel === "very-high") {
      recommendations.push("Learn and practice stress management techniques")
      recommendations.push("Consider mindfulness meditation or deep breathing exercises")
    }

    // Social support
    if (data.sleep.socialSupport === "none" || data.sleep.socialSupport === "minimal") {
      recommendations.push("Reach out to friends, family, or community groups")
      recommendations.push("Consider joining support groups or social activities")
    }

    return recommendations
  }

  private getCulturalConsiderations(region: string): string[] {
    const considerations: string[] = []

    switch (region) {
      case "western":
      case "central":
      case "southern":
      case "northern":
      case "eastern":
      case "northwest":
      case "northcentral":
      case "uva":
      case "sabaragamuwa":
        considerations.push("Mental health support is available in local languages (Sinhala/Tamil)")
        considerations.push("Consider speaking with religious or community leaders you trust")
        considerations.push("Family involvement in treatment decisions is culturally appropriate")
        considerations.push("Traditional healing practices can complement modern treatment")
        break
      default:
        considerations.push("Seek culturally competent mental health professionals")
        considerations.push("Consider cultural factors in treatment approaches")
    }

    return considerations
  }

  private getEmergencyResources(region: string): EmergencyResource[] {
    const resources: EmergencyResource[] = [
      {
        name: "Samaritans Lanka",
        phone: "0717 171 171",
        description: "Free confidential emotional support",
        available24h: true,
      },
      {
        name: "Sumithrayo",
        phone: "0112 682 535",
        description: "Suicide prevention and emotional support",
        available24h: true,
      },
      {
        name: "Emergency Services",
        phone: "1990",
        description: "Medical emergency services",
        available24h: true,
      },
      {
        name: "National Hospital Emergency",
        phone: "0112 691 111",
        description: "Medical emergency and psychiatric services",
        available24h: true,
      },
    ]

    return resources
  }

  public async assessRisk(data: AssessmentData): Promise<AssessmentResult> {
    try {
      const phq9Score = this.calculatePHQ9Score(data.phq9)
      const riskLevel = this.getRiskLevel(phq9Score)
      const riskFactors = this.analyzeRiskFactors(data)
      const protectiveFactors = this.analyzeProtectiveFactors(data)
      const recommendations = this.generateRecommendations(data, riskLevel)
      const culturalConsiderations = this.getCulturalConsiderations(data.demographics.region)
      const emergencyResources = this.getEmergencyResources(data.demographics.region)

      // Calculate confidence score based on completion of assessment
      let confidenceScore = 0.7 // Base confidence

      // Increase confidence based on data completeness
      const demographicComplete = Object.values(data.demographics).filter((v) => v).length / 5
      const phq9Complete = Object.values(data.phq9).filter((v) => v !== null).length / 9
      const sleepComplete = Object.values(data.sleep).filter((v) => v).length / 6

      confidenceScore += (demographicComplete + phq9Complete + sleepComplete) * 0.1
      confidenceScore = Math.min(confidenceScore, 0.95) // Cap at 95%

      return {
        riskLevel,
        phq9Score,
        riskFactors,
        protectiveFactors,
        recommendations,
        culturalConsiderations,
        emergencyResources,
        confidenceScore: Math.round(confidenceScore * 100) / 100,
      }
    } catch (error) {
      console.error("Error in AI assessment:", error)
      throw new Error("Assessment processing failed")
    }
  }
}
