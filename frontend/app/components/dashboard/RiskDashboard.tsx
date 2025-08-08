"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertTriangle, Phone, Brain, TrendingUp, Clock } from "lucide-react"
import BiometricRadarChart from "./BiometricRadarChart"
import PreventionPlan from "./PreventionPlan"
import InsightsAccordion from "./InsightsAccordion"
import type { PyTorchResults } from "@/lib/types"

interface RiskDashboardProps {
  results: PyTorchResults
}

export default function RiskDashboard({ results }: RiskDashboardProps) {
  const [showEmergencyCard, setShowEmergencyCard] = useState(results.risk_level === "high")

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case "low":
        return "bg-green-100 text-green-800 border-green-200"
      case "moderate":
        return "bg-yellow-100 text-yellow-800 border-yellow-200"
      case "high":
        return "bg-red-100 text-red-800 border-red-200"
      default:
        return "bg-gray-100 text-gray-800 border-gray-200"
    }
  }

  const getRiskIcon = (level: string) => {
    switch (level) {
      case "high":
        return <AlertTriangle className="w-5 h-5" />
      case "moderate":
        return <Clock className="w-5 h-5" />
      default:
        return <TrendingUp className="w-5 h-5" />
    }
  }

  return (
    <div className="space-y-6">
      {/* Risk Level Header */}
      <Card className={`border-2 ${getRiskLevelColor(results.risk_level)}`}>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {getRiskIcon(results.risk_level)}
              <div>
                <h2 className="text-2xl font-bold capitalize">{results.risk_level} Risk Level</h2>
                <p className="text-sm opacity-80">
                  Confidence: {Math.round(results.confidence * 100)}% • Analysis completed at{" "}
                  {new Date().toLocaleTimeString()}
                </p>
              </div>
            </div>
            <Badge variant="outline" className="text-lg px-4 py-2">
              <Brain className="w-4 h-4 mr-2" />
              AI Analysis
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Emergency Card for High Risk */}
      {showEmergencyCard && results.risk_level === "high" && (
        <Alert className="border-red-500 bg-red-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <div className="flex items-center justify-between">
              <div>
                <strong>Immediate Support Available</strong>
                <p className="mt-1">Your assessment indicates you may benefit from immediate professional support.</p>
              </div>
              <div className="flex gap-2">
                <Button size="sm" className="bg-red-600 hover:bg-red-700">
                  <Phone className="w-4 h-4 mr-2" />
                  Call 988
                </Button>
                <Button size="sm" variant="outline" onClick={() => setShowEmergencyCard(false)}>
                  Dismiss
                </Button>
              </div>
            </div>
          </AlertDescription>
        </Alert>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Biometric Radar Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Mental Health Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <BiometricRadarChart
              metrics={results.biometric_scores}
              thresholds={{
                sleep: 7,
                mood: 6,
                social: 6,
                stress: 4,
                energy: 6,
              }}
            />
          </CardContent>
        </Card>

        {/* Key Risk Factors */}
        <Card>
          <CardHeader>
            <CardTitle>Risk Factor Analysis</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {Object.entries(results.key_factors).map(([factor, data]) => (
              <div key={factor} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <h4 className="font-medium capitalize">{factor.replace("_", " ")}</h4>
                  <p className="text-sm text-gray-600">{data.value}</p>
                </div>
                <Badge variant={data.impact === "high" ? "destructive" : "secondary"} className="capitalize">
                  {data.impact} Impact
                </Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Prevention Plan */}
      <PreventionPlan
        interventions={results.interventions}
        riskLevel={results.risk_level}
        recommendations={results.recommendations}
      />

      {/* Clinical Insights */}
      <InsightsAccordion
        title="Clinical Insights & Methodology"
        confidence={results.confidence}
        keyFactors={results.key_factors}
        riskLevel={results.risk_level}
      />
    </div>
  )
}
