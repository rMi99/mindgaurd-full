import type { AssessmentHistory, HistoricalTrend } from "./types"

export class HistoricalAnalysis {
  public analyzeTrends(history: AssessmentHistory[]): HistoricalTrend {
    const sortedHistory = history.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())

    return {
      overallTrend: this.calculateOverallTrend(sortedHistory),
      phq9Trend: this.calculatePHQ9Trend(sortedHistory),
      sleepTrend: this.calculateSleepTrend(sortedHistory),
      insights: this.generateInsights(sortedHistory),
      recommendations: this.generateRecommendations(sortedHistory),
      correlations: this.findCorrelations(sortedHistory),
    }
  }

  private calculateOverallTrend(history: AssessmentHistory[]): "improving" | "stable" | "declining" {
    if (history.length < 3) return "stable"

    const recent = history.slice(-3)
    const older = history.slice(-6, -3)

    if (older.length === 0) return "stable"

    const recentAvg = recent.reduce((sum, h) => sum + h.phq9Score, 0) / recent.length
    const olderAvg = older.reduce((sum, h) => sum + h.phq9Score, 0) / older.length

    const difference = recentAvg - olderAvg

    if (difference < -2) return "improving"
    if (difference > 2) return "declining"
    return "stable"
  }

  private calculatePHQ9Trend(history: AssessmentHistory[]): number {
    if (history.length < 2) return 0

    const recent = history.slice(-5)
    const scores = recent.map((h) => h.phq9Score)

    // Simple linear regression slope
    const n = scores.length
    const sumX = (n * (n - 1)) / 2
    const sumY = scores.reduce((sum, score) => sum + score, 0)
    const sumXY = scores.reduce((sum, score, index) => sum + score * index, 0)
    const sumXX = (n * (n - 1) * (2 * n - 1)) / 6

    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
  }

  private calculateSleepTrend(history: AssessmentHistory[]): "improving" | "stable" | "declining" {
    if (history.length < 3) return "stable"

    const recent = history.slice(-3)
    const older = history.slice(-6, -3)

    if (older.length === 0) return "stable"

    const recentAvg = recent.reduce((sum, h) => sum + h.sleepHoursNumeric, 0) / recent.length
    const olderAvg = older.reduce((sum, h) => sum + h.sleepHoursNumeric, 0) / older.length

    const difference = recentAvg - olderAvg

    if (difference > 0.5) return "improving"
    if (difference < -0.5) return "declining"
    return "stable"
  }

  private generateInsights(history: AssessmentHistory[]): string[] {
    const insights: string[] = []

    // Sleep-mood correlation
    const sleepMoodCorrelation = this.calculateSleepMoodCorrelation(history)
    if (sleepMoodCorrelation < -0.3) {
      insights.push("Your mood tends to improve when you get more sleep. Consider prioritizing sleep hygiene.")
    } else if (sleepMoodCorrelation > 0.3) {
      insights.push(
        "Interestingly, your mood doesn't seem strongly correlated with sleep duration. Other factors may be more influential.",
      )
    }

    // Trend insights
    const trend = this.calculateOverallTrend(history)
    if (trend === "improving") {
      insights.push(
        "Great news! Your mental health scores have been improving over time. Keep up the positive changes.",
      )
    } else if (trend === "declining") {
      insights.push("Your recent scores suggest some challenges. Consider reaching out for additional support.")
    }

    // Consistency insights
    const consistency = this.calculateConsistency(history)
    if (consistency < 0.3) {
      insights.push("Your scores show high variability. Identifying triggers and patterns could be helpful.")
    } else if (consistency > 0.7) {
      insights.push("Your scores are quite consistent, which suggests stable patterns in your mental health.")
    }

    return insights
  }

  private generateRecommendations(history: AssessmentHistory[]): string[] {
    const recommendations: string[] = []
    const latest = history[history.length - 1]

    // Sleep recommendations
    if (latest.sleepHoursNumeric < 6) {
      recommendations.push("Aim for 7-9 hours of sleep per night. Consider establishing a consistent bedtime routine.")
    }

    // Exercise recommendations
    if (latest.exerciseFrequency === "never" || latest.exerciseFrequency === "1-2") {
      recommendations.push(
        "Regular physical activity can significantly improve mood. Start with 30 minutes of walking 3 times per week.",
      )
    }

    // Social support recommendations
    if (latest.socialSupport === "minimal" || latest.socialSupport === "none") {
      recommendations.push(
        "Building social connections is crucial for mental health. Consider joining groups or activities that interest you.",
      )
    }

    // Stress management
    if (latest.stressLevel === "high" || latest.stressLevel === "very-high") {
      recommendations.push(
        "High stress levels can impact mental health. Try stress-reduction techniques like meditation or deep breathing.",
      )
    }

    // Professional help
    if (latest.phq9Score > 14) {
      recommendations.push(
        "Your scores suggest you might benefit from professional support. Consider speaking with a healthcare provider.",
      )
    }

    return recommendations
  }

  private findCorrelations(history: AssessmentHistory[]): Record<string, number> {
    return {
      sleepMood: this.calculateSleepMoodCorrelation(history),
      exerciseMood: this.calculateExerciseMoodCorrelation(history),
      stressMood: this.calculateStressMoodCorrelation(history),
    }
  }

  private calculateSleepMoodCorrelation(history: AssessmentHistory[]): number {
    if (history.length < 3) return 0

    const sleepHours = history.map((h) => h.sleepHoursNumeric)
    const moodScores = history.map((h) => -h.phq9Score) // Negative because lower PHQ-9 is better

    return this.pearsonCorrelation(sleepHours, moodScores)
  }

  private calculateExerciseMoodCorrelation(history: AssessmentHistory[]): number {
    if (history.length < 3) return 0

    const exerciseScores = history.map((h) => this.exerciseToNumeric(h.exerciseFrequency))
    const moodScores = history.map((h) => -h.phq9Score)

    return this.pearsonCorrelation(exerciseScores, moodScores)
  }

  private calculateStressMoodCorrelation(history: AssessmentHistory[]): number {
    if (history.length < 3) return 0

    const stressScores = history.map((h) => this.stressToNumeric(h.stressLevel))
    const moodScores = history.map((h) => -h.phq9Score)

    return this.pearsonCorrelation(stressScores, moodScores)
  }

  private exerciseToNumeric(exercise: string): number {
    switch (exercise) {
      case "never":
        return 0
      case "1-2":
        return 1.5
      case "3-4":
        return 3.5
      case "5-6":
        return 5.5
      case "daily":
        return 7
      default:
        return 0
    }
  }

  private stressToNumeric(stress: string): number {
    switch (stress) {
      case "very-low":
        return 1
      case "low":
        return 2
      case "moderate":
        return 3
      case "high":
        return 4
      case "very-high":
        return 5
      default:
        return 3
    }
  }

  private calculateConsistency(history: AssessmentHistory[]): number {
    if (history.length < 3) return 0

    const scores = history.map((h) => h.phq9Score)
    const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length
    const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length
    const standardDeviation = Math.sqrt(variance)

    // Return inverse of coefficient of variation (lower variation = higher consistency)
    return mean > 0 ? 1 - standardDeviation / mean : 0
  }

  private pearsonCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length < 2) return 0

    const n = x.length
    const sumX = x.reduce((sum, val) => sum + val, 0)
    const sumY = y.reduce((sum, val) => sum + val, 0)
    const sumXY = x.reduce((sum, val, i) => sum + val * y[i], 0)
    const sumXX = x.reduce((sum, val) => sum + val * val, 0)
    const sumYY = y.reduce((sum, val) => sum + val * val, 0)

    const numerator = n * sumXY - sumX * sumY
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY))

    return denominator === 0 ? 0 : numerator / denominator
  }
}
