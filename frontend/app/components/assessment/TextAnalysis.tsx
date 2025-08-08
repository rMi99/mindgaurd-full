"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { MessageSquare, Activity, Users, Zap, TrendingUp, TrendingDown } from "lucide-react"
import type { BehavioralData } from "@/lib/types"

interface TextAnalysisProps {
  data: BehavioralData
  onChange: (data: BehavioralData) => void
  multilingual: boolean
  placeholder: string
  sentimentAnalysis: boolean
}

export default function TextAnalysis({ data, onChange, placeholder, sentimentAnalysis }: TextAnalysisProps) {
  const [wordCount, setWordCount] = useState(0)
  const [sentimentProcessing, setSentimentProcessing] = useState(false)

  useEffect(() => {
    const words = data.moodDescription
      .trim()
      .split(/\s+/)
      .filter((word) => word.length > 0)
    setWordCount(words.length)

    if (sentimentAnalysis && data.moodDescription.length > 20) {
      analyzeSentiment(data.moodDescription)
    }
  }, [data.moodDescription])

  const analyzeSentiment = async (text: string) => {
    setSentimentProcessing(true)
    try {
      // Mock sentiment analysis - in production, this would call your NLP API
      const response = await fetch("/api/sentiment-analysis", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })

      if (response.ok) {
        const result = await response.json()
        onChange({ ...data, sentimentScore: result.score })
      } else {
        // Fallback: simple sentiment analysis
        const positiveWords = ["good", "great", "happy", "better", "positive", "hopeful", "calm", "peaceful"]
        const negativeWords = ["bad", "terrible", "sad", "worse", "negative", "hopeless", "anxious", "stressed"]

        const words = text.toLowerCase().split(/\s+/)
        const positiveCount = words.filter((word) => positiveWords.includes(word)).length
        const negativeCount = words.filter((word) => negativeWords.includes(word)).length

        const score = (positiveCount - negativeCount) / Math.max(words.length / 10, 1)
        onChange({ ...data, sentimentScore: Math.max(-1, Math.min(1, score)) })
      }
    } catch (error) {
      console.error("Sentiment analysis failed:", error)
    } finally {
      setSentimentProcessing(false)
    }
  }

  const getSentimentColor = (score: number) => {
    if (score > 0.3) return "text-green-600"
    if (score < -0.3) return "text-red-600"
    return "text-yellow-600"
  }

  const getSentimentLabel = (score: number) => {
    if (score > 0.5) return "Very Positive"
    if (score > 0.2) return "Positive"
    if (score > -0.2) return "Neutral"
    if (score > -0.5) return "Negative"
    return "Very Negative"
  }

  return (
    <div className="space-y-6">
      {/* Mood Description */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5" />
            Mood & Feelings Description
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Textarea
              placeholder={placeholder}
              value={data.moodDescription}
              onChange={(e) => onChange({ ...data, moodDescription: e.target.value })}
              className="min-h-[120px] resize-none"
              maxLength={1000}
            />
            <div className="flex justify-between items-center mt-2 text-sm text-gray-500">
              <span>{wordCount} words</span>
              <span>{data.moodDescription.length}/1000 characters</span>
            </div>
          </div>

          {sentimentAnalysis && data.moodDescription.length > 20 && (
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">Sentiment Analysis</span>
                {sentimentProcessing ? (
                  <Badge variant="outline">Processing...</Badge>
                ) : (
                  <Badge className={getSentimentColor(data.sentimentScore)}>
                    {getSentimentLabel(data.sentimentScore)}
                  </Badge>
                )}
              </div>

              <div className="flex items-center gap-2">
                <TrendingDown className="w-4 h-4 text-red-500" />
                <Progress value={((data.sentimentScore + 1) / 2) * 100} className="flex-1 h-2" />
                <TrendingUp className="w-4 h-4 text-green-500" />
              </div>

              <p className="text-xs text-gray-600 mt-2">AI analysis of emotional tone in your description</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Behavioral Metrics */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5" />
              Stress Level
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Current stress level</span>
                <Badge variant="outline">{data.stressLevel}/5</Badge>
              </div>
              <Slider
                value={[data.stressLevel]}
                onValueChange={(value) => onChange({ ...data, stressLevel: value[0] })}
                max={5}
                min={1}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Very Low</span>
                <span>Moderate</span>
                <span>Very High</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="w-5 h-5" />
              Social Connections
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Social interaction quality</span>
                <Badge variant="outline">{data.socialConnections}/5</Badge>
              </div>
              <Slider
                value={[data.socialConnections]}
                onValueChange={(value) => onChange({ ...data, socialConnections: value[0] })}
                max={5}
                min={1}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Isolated</span>
                <span>Moderate</span>
                <span>Very Connected</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5" />
              Physical Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Exercise frequency</span>
                <Badge variant="outline">{data.physicalActivity}/5</Badge>
              </div>
              <Slider
                value={[data.physicalActivity]}
                onValueChange={(value) => onChange({ ...data, physicalActivity: value[0] })}
                max={5}
                min={1}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500">
                <span>Sedentary</span>
                <span>Moderate</span>
                <span>Very Active</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Validation Messages */}
      {data.moodDescription.length < 10 && (
        <Card className="border-orange-200 bg-orange-50">
          <CardContent className="p-4">
            <p className="text-sm text-orange-700">
              Please provide a more detailed description of your mood and feelings (minimum 10 characters).
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
