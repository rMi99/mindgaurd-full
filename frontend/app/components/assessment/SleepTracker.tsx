"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Moon, Sun, Clock, TrendingUp } from "lucide-react"
import type { SleepData } from "@/lib/types"

interface SleepTrackerProps {
  type: "interactive_calendar"
  data: SleepData
  onChange: (data: SleepData) => void
  defaultHours: number
}

export default function SleepTracker({ data, onChange, defaultHours }: SleepTrackerProps) {
  const [selectedDay, setSelectedDay] = useState(0)

  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
  const qualityLabels = ["Very Poor", "Poor", "Fair", "Good", "Excellent"]

  const updateWeeklyPattern = (dayIndex: number, hours: number) => {
    const newPattern = [...data.weeklyPattern]
    newPattern[dayIndex] = hours
    const averageHours = newPattern.reduce((sum, h) => sum + h, 0) / 7

    onChange({
      ...data,
      weeklyPattern: newPattern,
      averageHours: Math.round(averageHours * 10) / 10,
    })
  }

  const getSleepQualityColor = (quality: number) => {
    if (quality <= 1) return "bg-red-100 text-red-800"
    if (quality <= 2) return "bg-orange-100 text-orange-800"
    if (quality <= 3) return "bg-yellow-100 text-yellow-800"
    if (quality <= 4) return "bg-green-100 text-green-800"
    return "bg-blue-100 text-blue-800"
  }

  const getHoursColor = (hours: number) => {
    if (hours < 6) return "text-red-600"
    if (hours < 7) return "text-orange-600"
    if (hours <= 9) return "text-green-600"
    return "text-blue-600"
  }

  return (
    <div className="space-y-6">
      {/* Sleep Overview */}
      <div className="grid md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <Moon className="w-8 h-8 text-blue-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">{data.averageHours}h</div>
            <div className="text-sm text-gray-600">Average Sleep</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Sun className="w-8 h-8 text-yellow-600 mx-auto mb-2" />
            <Badge className={getSleepQualityColor(data.quality)}>{qualityLabels[data.quality - 1] || "Fair"}</Badge>
            <div className="text-sm text-gray-600 mt-1">Sleep Quality</div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <TrendingUp className="w-8 h-8 text-green-600 mx-auto mb-2" />
            <div className="text-2xl font-bold text-gray-900">{data.consistency}/5</div>
            <div className="text-sm text-gray-600">Consistency</div>
          </CardContent>
        </Card>
      </div>

      {/* Weekly Sleep Pattern */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="w-5 h-5" />
            Weekly Sleep Pattern
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-7 gap-2">
            {days.map((day, index) => (
              <div
                key={day}
                className={`p-3 text-center rounded-lg border-2 cursor-pointer transition-all ${
                  selectedDay === index ? "border-blue-500 bg-blue-50" : "border-gray-200 hover:border-gray-300"
                }`}
                onClick={() => setSelectedDay(index)}
              >
                <div className="text-sm font-medium text-gray-700">{day}</div>
                <div className={`text-lg font-bold ${getHoursColor(data.weeklyPattern[index])}`}>
                  {data.weeklyPattern[index]}h
                </div>
              </div>
            ))}
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Sleep hours for {days[selectedDay]}: {data.weeklyPattern[selectedDay]}h
              </label>
              <Slider
                value={[data.weeklyPattern[selectedDay]]}
                onValueChange={(value) => updateWeeklyPattern(selectedDay, value[0])}
                max={12}
                min={0}
                step={0.5}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>0h</span>
                <span>6h</span>
                <span>9h</span>
                <span>12h</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Sleep Quality Assessment */}
      <Card>
        <CardHeader>
          <CardTitle>Sleep Quality Assessment</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Overall Sleep Quality: {qualityLabels[data.quality - 1] || "Fair"}
            </label>
            <Slider
              value={[data.quality]}
              onValueChange={(value) => onChange({ ...data, quality: value[0] })}
              max={5}
              min={1}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              {qualityLabels.map((label, index) => (
                <span key={index} className={data.quality === index + 1 ? "font-medium text-blue-600" : ""}>
                  {label}
                </span>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Sleep Schedule Consistency: {data.consistency}/5
            </label>
            <Slider
              value={[data.consistency]}
              onValueChange={(value) => onChange({ ...data, consistency: value[0] })}
              max={5}
              min={1}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Very Irregular</span>
              <span>Somewhat Consistent</span>
              <span>Very Consistent</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Sleep Recommendations */}
      {data.averageHours < 7 && (
        <Card className="border-orange-200 bg-orange-50">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <Moon className="w-5 h-5 text-orange-600" />
              <div>
                <h4 className="font-medium text-orange-800">Sleep Optimization Needed</h4>
                <p className="text-sm text-orange-700">
                  Adults need 7-9 hours of sleep per night for optimal mental health. Consider establishing a consistent
                  bedtime routine.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
