"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Clock, Play, Heart, Zap } from "lucide-react"
import EmergencyCard from "./EmergencyCard"
import MicroActions from "./MicroActions"
import type { PyTorchResults } from "@/lib/types"

interface PreventionPlanProps {
  interventions: PyTorchResults["interventions"]
  riskLevel: string
  recommendations: string[]
}

export default function PreventionPlan({ interventions, riskLevel, recommendations }: PreventionPlanProps) {
  const [completedActions, setCompletedActions] = useState<Set<string>>(new Set())
  const [activeBreathing, setActiveBreathing] = useState(false)

  const handleActionComplete = (actionId: string) => {
    setCompletedActions((prev) => new Set([...prev, actionId]))
  }

  const microActions = [
    {
      id: "hydration",
      icon: "💧",
      text: "Drink a glass of water",
      reason: "Hydration affects mood and cognitive function",
      duration: "30 seconds",
      category: "immediate",
    },
    {
      id: "breathing",
      icon: "🌿",
      text: "Try 4-4-6 breathing",
      reason: "Activates parasympathetic nervous system",
      duration: "2 minutes",
      category: "immediate",
      onClick: () => setActiveBreathing(true),
    },
    {
      id: "sunlight",
      icon: "☀️",
      text: "Get 5 minutes of sunlight",
      reason: "Natural light regulates circadian rhythm",
      duration: "5 minutes",
      category: "immediate",
    },
    {
      id: "movement",
      icon: "🚶",
      text: "Take a short walk",
      reason: "Physical activity releases endorphins",
      duration: "10 minutes",
      category: "short-term",
    },
  ]

  const completionRate = (completedActions.size / microActions.length) * 100

  return (
    <div className="space-y-6">
      {/* Emergency Card */}
      {riskLevel === "high" && (
        <EmergencyCard
          visible={true}
          contacts={[
            { name: "National Suicide Prevention Lifeline", number: "988", available24h: true },
            { name: "Crisis Text Line", number: "741741", available24h: true, isText: true },
            { name: "Emergency Services", number: "911", available24h: true },
          ]}
        />
      )}

      {/* Immediate Actions */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-5 h-5 text-blue-600" />
              Immediate Actions
            </CardTitle>
            <div className="flex items-center gap-2">
              <Progress value={completionRate} className="w-20 h-2" />
              <span className="text-sm text-gray-600">{Math.round(completionRate)}%</span>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <MicroActions
            actions={microActions.filter((action) => action.category === "immediate")}
            completedActions={completedActions}
            onActionComplete={handleActionComplete}
          />
        </CardContent>
      </Card>

      {/* AI-Recommended Interventions */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-green-600" />
              Immediate Interventions
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {interventions.immediate.map((intervention, index) => (
              <div key={index} className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-green-900">{intervention.title}</h4>
                  <Badge variant="outline" className="text-green-700">
                    {intervention.duration}
                  </Badge>
                </div>
                <p className="text-sm text-green-800 mb-3">{intervention.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-green-600">Reason: {intervention.reason}</span>
                  <Button
                    size="sm"
                    className="bg-green-600 hover:bg-green-700"
                    onClick={() => {
                      if (intervention.type === "breathing") {
                        setActiveBreathing(true)
                      }
                    }}
                  >
                    <Play className="w-4 h-4 mr-1" />
                    Start
                  </Button>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Heart className="w-5 h-5 text-purple-600" />
              Long-term Strategy
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {interventions.longterm.map((intervention, index) => (
              <div key={index} className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                <h4 className="font-medium text-purple-900 mb-2">{intervention.title}</h4>
                <p className="text-sm text-purple-800 mb-3">{intervention.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-purple-600">Plan: {intervention.plan}</span>
                  <Button size="sm" variant="outline" className="border-purple-300 text-purple-700 bg-transparent">
                    Learn More
                  </Button>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Personalized Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle>Personalized Recommendations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-4">
            {recommendations.map((recommendation, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-blue-50 rounded-lg">
                <div className="flex-shrink-0 w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center mt-0.5">
                  <span className="text-xs font-medium text-blue-600">{index + 1}</span>
                </div>
                <p className="text-sm text-blue-800">{recommendation}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Breathing Exercise Modal */}
      {activeBreathing && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md mx-4">
            <CardHeader>
              <CardTitle className="text-center">4-4-6 Breathing Exercise</CardTitle>
            </CardHeader>
            <CardContent className="text-center space-y-4">
              <div className="w-32 h-32 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
                <div className="w-24 h-24 bg-blue-200 rounded-full animate-pulse"></div>
              </div>
              <div className="space-y-2">
                <p className="text-sm text-gray-600">Breathe in for 4 seconds</p>
                <p className="text-sm text-gray-600">Hold for 4 seconds</p>
                <p className="text-sm text-gray-600">Breathe out for 6 seconds</p>
              </div>
              <Button onClick={() => setActiveBreathing(false)} className="w-full">
                Complete Exercise
              </Button>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}
