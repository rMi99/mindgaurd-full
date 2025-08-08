"use client"

import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Info, AlertTriangle } from "lucide-react"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface SliderInputProps {
  title: string
  subtitle?: string
  questions: string[]
  range: [number, number]
  labels: string[]
  values: number[]
  onChange: (values: number[]) => void
  tooltips?: {
    clinical?: string
    scoring?: string
  }
}

export default function SliderInput({
  title,
  subtitle,
  questions,
  range,
  labels,
  values,
  onChange,
  tooltips,
}: SliderInputProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const totalScore = values.reduce((sum, score) => sum + score, 0)
  const maxScore = questions.length * range[1]

  const handleValueChange = (questionIndex: number, newValue: number[]) => {
    const newValues = [...values]
    newValues[questionIndex] = newValue[0]
    onChange(newValues)
  }

  const getRiskLevel = (score: number) => {
    if (score <= 4) return { level: "Minimal", color: "bg-green-100 text-green-800" }
    if (score <= 9) return { level: "Mild", color: "bg-yellow-100 text-yellow-800" }
    if (score <= 14) return { level: "Moderate", color: "bg-orange-100 text-orange-800" }
    if (score <= 19) return { level: "Moderately Severe", color: "bg-red-100 text-red-800" }
    return { level: "Severe", color: "bg-red-200 text-red-900" }
  }

  const riskLevel = getRiskLevel(totalScore)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
          {subtitle && <p className="text-sm text-gray-600 mt-1">{subtitle}</p>}
        </div>
        <div className="flex items-center gap-4">
          <Badge className={riskLevel.color}>
            {riskLevel.level}: {totalScore}/{maxScore}
          </Badge>
          {tooltips && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger>
                  <Info className="w-4 h-4 text-gray-400" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  <div className="space-y-2">
                    {tooltips.clinical && (
                      <p>
                        <strong>Clinical:</strong> {tooltips.clinical}
                      </p>
                    )}
                    {tooltips.scoring && (
                      <p>
                        <strong>Scoring:</strong> {tooltips.scoring}
                      </p>
                    )}
                  </div>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      </div>

      {/* Questions */}
      <div className="space-y-6">
        {questions.map((question, index) => (
          <Card
            key={index}
            className={`transition-all duration-200 ${
              index === currentQuestion ? "ring-2 ring-blue-500 bg-blue-50" : ""
            }`}
          >
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900 mb-2">
                      {index + 1}. {question}
                    </h4>
                    {index === 8 && values[8] > 0 && (
                      <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-lg">
                        <AlertTriangle className="w-4 h-4 text-red-600" />
                        <span className="text-sm text-red-800">
                          If you're having thoughts of self-harm, please reach out for help immediately.
                        </span>
                      </div>
                    )}
                  </div>
                  <Badge variant="outline" className="ml-4">
                    {values[index]}/3
                  </Badge>
                </div>

                <div className="space-y-3">
                  <Slider
                    value={[values[index]]}
                    onValueChange={(value) => handleValueChange(index, value)}
                    max={range[1]}
                    min={range[0]}
                    step={1}
                    className="w-full"
                    onValueCommit={() => setCurrentQuestion(index)}
                  />

                  <div className="flex justify-between text-xs text-gray-500">
                    {labels.map((label, labelIndex) => (
                      <span
                        key={labelIndex}
                        className={`${values[index] === labelIndex ? "font-medium text-blue-600" : ""}`}
                      >
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Crisis Resources */}
      {totalScore > 14 && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-red-600" />
              <div>
                <h4 className="font-medium text-red-800">Support Available</h4>
                <p className="text-sm text-red-700">
                  Your responses suggest you might benefit from professional support. Crisis helplines: 988 (US), 0717
                  171 171 (Sri Lanka)
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
