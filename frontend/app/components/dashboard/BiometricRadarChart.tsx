"use client"

import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from "recharts"

interface BiometricRadarChartProps {
  metrics: {
    sleep: number
    mood: number
    social: number
    stress: number
    energy: number
  }
  thresholds: {
    sleep: number
    mood: number
    social: number
    stress: number
    energy: number
  }
}

export default function BiometricRadarChart({ metrics, thresholds }: BiometricRadarChartProps) {
  const data = [
    {
      metric: "Sleep Quality",
      current: metrics.sleep,
      optimal: thresholds.sleep,
      fullMark: 10,
    },
    {
      metric: "Mood Stability",
      current: metrics.mood,
      optimal: thresholds.mood,
      fullMark: 10,
    },
    {
      metric: "Social Connection",
      current: metrics.social,
      optimal: thresholds.social,
      fullMark: 10,
    },
    {
      metric: "Stress Level",
      current: 10 - metrics.stress, // Invert stress (lower is better)
      optimal: 10 - thresholds.stress,
      fullMark: 10,
    },
    {
      metric: "Energy Level",
      current: metrics.energy,
      optimal: thresholds.energy,
      fullMark: 10,
    },
  ]

  return (
    <div className="w-full h-80">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
          <PolarGrid gridType="polygon" />
          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12, fill: "#6B7280" }} className="text-xs" />
          <PolarRadiusAxis angle={90} domain={[0, 10]} tick={{ fontSize: 10, fill: "#9CA3AF" }} tickCount={6} />
          <Radar name="Current" dataKey="current" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} strokeWidth={2} />
          <Radar
            name="Optimal Range"
            dataKey="optimal"
            stroke="#10B981"
            fill="#10B981"
            fillOpacity={0.1}
            strokeWidth={2}
            strokeDasharray="5 5"
          />
          <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "20px" }} />
        </RadarChart>
      </ResponsiveContainer>

      <div className="mt-4 text-center">
        <p className="text-sm text-gray-600">
          Radar chart showing your current mental health metrics compared to optimal clinical baselines
        </p>
      </div>
    </div>
  )
}
