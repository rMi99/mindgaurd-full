import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const { text } = await request.json()

    if (!text || text.length < 10) {
      return NextResponse.json({ error: "Text too short for analysis" }, { status: 400 })
    }

    // Mock sentiment analysis - in production, this would use a real NLP service
    const sentimentScore = await analyzeSentiment(text)

    return NextResponse.json({ score: sentimentScore })
  } catch (error) {
    console.error("Sentiment analysis error:", error)
    return NextResponse.json({ error: "Sentiment analysis failed" }, { status: 500 })
  }
}

async function analyzeSentiment(text: string): Promise<number> {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 1000))

  // Simple sentiment analysis using keyword matching
  const positiveWords = [
    "good",
    "great",
    "happy",
    "better",
    "positive",
    "hopeful",
    "calm",
    "peaceful",
    "excited",
    "grateful",
    "love",
    "joy",
    "wonderful",
    "amazing",
    "excellent",
    "fantastic",
    "brilliant",
    "awesome",
    "perfect",
    "beautiful",
    "successful",
    "confident",
    "optimistic",
    "cheerful",
    "delighted",
    "pleased",
    "satisfied",
  ]

  const negativeWords = [
    "bad",
    "terrible",
    "sad",
    "worse",
    "negative",
    "hopeless",
    "anxious",
    "stressed",
    "depressed",
    "angry",
    "frustrated",
    "worried",
    "scared",
    "afraid",
    "lonely",
    "tired",
    "exhausted",
    "overwhelmed",
    "disappointed",
    "upset",
    "hurt",
    "pain",
    "difficult",
    "hard",
    "struggle",
    "problem",
    "issue",
    "trouble",
    "crisis",
  ]

  const words = text.toLowerCase().split(/\s+/)
  const positiveCount = words.filter((word) => positiveWords.includes(word)).length
  const negativeCount = words.filter((word) => negativeWords.includes(word)).length

  // Calculate sentiment score (-1 to 1)
  const totalSentimentWords = positiveCount + negativeCount
  if (totalSentimentWords === 0) return 0

  const rawScore = (positiveCount - negativeCount) / totalSentimentWords

  // Normalize and add some randomness for realism
  const normalizedScore = rawScore * 0.8 + (Math.random() - 0.5) * 0.2

  return Math.max(-1, Math.min(1, normalizedScore))
}
