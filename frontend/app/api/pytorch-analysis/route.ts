import { type NextRequest, NextResponse } from "next/server"
import type { AssessmentData, PyTorchResults } from "@/lib/types"

export async function POST(request: NextRequest) {
  try {
    const assessmentData: AssessmentData = await request.json()

    // In production, this would call your actual PyTorch backend
    // For now, we'll simulate the analysis with realistic mock data
    const mockResults = await simulatePyTorchAnalysis(assessmentData)

    return NextResponse.json(mockResults)
  } catch (error) {
    console.error("PyTorch analysis error:", error)
    return NextResponse.json({ error: "Analysis processing failed" }, { status: 500 })
  }
}

async function simulatePyTorchAnalysis(data: AssessmentData): Promise<PyTorchResults> {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 2000))

  // Calculate PHQ-9 total score
  const phq9Total = data.phq9.scores.reduce((sum, score) => sum + score, 0)

  // Determine risk level based on PHQ-9 and other factors
  let riskLevel: "low" | "moderate" | "high" = "low"
  if (phq9Total >= 15 || data.phq9.scores[8] > 0) {
    riskLevel = "high"
  } else if (phq9Total >= 10 || data.sleep.averageHours < 6) {
    riskLevel = "moderate"
  }

  // Calculate biometric scores (0-10 scale)
  const sleepScore = Math.max(0, Math.min(10, (data.sleep.averageHours / 8) * 10))
  const moodScore = Math.max(0, 10 - (phq9Total / 27) * 10)
  const socialScore = data.behavioral.socialConnections * 2
  const stressScore = data.behavioral.stressLevel * 2
  const energyScore = Math.max(0, 10 - data.behavioral.stressLevel * 2)

  // Generate key factors
  const keyFactors: Record<string, { value: string; impact: string }> = {}

  if (data.sleep.averageHours < 7) {
    keyFactors.sleep_deficit = {
      value: `${data.sleep.averageHours}h/night`,
      impact: data.sleep.averageHours < 6 ? "high" : "moderate",
    }
  }

  if (phq9Total > 5) {
    keyFactors.phq9_score = {
      value: `${phq9Total}/27`,
      impact: phq9Total > 14 ? "high" : "moderate",
    }
  }

  if (data.behavioral.socialConnections <= 2) {
    keyFactors.social_isolation = {
      value: `${data.behavioral.socialConnections}/5 social connection score`,
      impact: "moderate",
    }
  }

  if (data.behavioral.stressLevel >= 4) {
    keyFactors.high_stress = {
      value: `${data.behavioral.stressLevel}/5 stress level`,
      impact: "high",
    }
  }

  // Generate interventions based on risk level and factors
  const interventions = {
    immediate: [
      {
        type: "breathing",
        duration: "5min",
        reason: "elevated_stress_indicators",
        title: "4-4-6 Breathing Exercise",
        description: "Controlled breathing to activate parasympathetic nervous system and reduce acute stress",
      },
    ],
    longterm: [
      {
        type: "sleep_hygiene",
        plan: "7-9_hour_sleep_schedule",
        title: "Sleep Optimization Program",
        description: "Establish consistent sleep-wake cycle to improve mood regulation and cognitive function",
      },
    ],
  }

  // Add crisis intervention for high risk
  if (riskLevel === "high") {
    interventions.immediate.unshift({
      type: "crisis_support",
      duration: "immediate",
      reason: "high_risk_indicators",
      title: "Crisis Support Resources",
      description: "Immediate access to professional crisis intervention and safety planning",
    })
  }

  // Generate personalized recommendations
  const recommendations = []

  if (data.sleep.averageHours < 7) {
    recommendations.push("Prioritize 7-9 hours of sleep nightly for optimal mental health")
  }

  if (data.behavioral.stressLevel >= 4) {
    recommendations.push("Practice daily stress-reduction techniques like meditation or deep breathing")
  }

  if (data.behavioral.socialConnections <= 2) {
    recommendations.push("Strengthen social connections through regular contact with friends or family")
  }

  if (data.behavioral.physicalActivity <= 2) {
    recommendations.push("Incorporate regular physical activity - even 30 minutes of walking can improve mood")
  }

  if (phq9Total > 10) {
    recommendations.push("Consider speaking with a healthcare provider about your mental health")
  }

  // Calculate confidence based on data completeness and consistency
  let confidence = 0.75 // Base confidence

  // Increase confidence for complete data
  if (data.phq9.scores.every((score) => score >= 0)) confidence += 0.1
  if (data.behavioral.moodDescription.length > 50) confidence += 0.05
  if (data.sleep.weeklyPattern.every((hours) => hours > 0)) confidence += 0.05

  // Decrease confidence for inconsistent data
  if (Math.abs(data.sleep.averageHours - data.sleep.weeklyPattern.reduce((a, b) => a + b, 0) / 7) > 1) {
    confidence -= 0.1
  }

  return {
    risk_level: riskLevel,
    confidence: Math.min(0.95, confidence),
    key_factors: keyFactors,
    interventions,
    biometric_scores: {
      sleep: Math.round(sleepScore * 10) / 10,
      mood: Math.round(moodScore * 10) / 10,
      social: Math.round(socialScore * 10) / 10,
      stress: Math.round(stressScore * 10) / 10,
      energy: Math.round(energyScore * 10) / 10,
    },
    recommendations,
  }
}
