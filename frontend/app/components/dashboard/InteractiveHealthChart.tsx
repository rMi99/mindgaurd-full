'use client';

import React, { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ComposedChart
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Calendar, TrendingUp, Activity, Heart, Brain, Zap } from 'lucide-react';

interface HealthDataPoint {
  date: string;
  sleep_hours: number;
  stress_level: number;
  exercise_frequency: number;
  mood_score: number;
  energy_level: number;
  social_connections: number;
  diet_quality: number;
}

interface ChartConfig {
  type: 'line' | 'area' | 'bar' | 'pie' | 'radar' | 'composed';
  title: string;
  description: string;
  icon: React.ReactNode;
}

interface InteractiveHealthChartProps {
  data: HealthDataPoint[];
  riskLevel?: 'low' | 'normal' | 'high';
  className?: string;
}

const COLORS = {
  low: '#10b981',
  normal: '#f59e0b',
  high: '#ef4444',
  primary: '#3b82f6',
  secondary: '#8b5cf6',
  accent: '#06b6d4'
};

const CHART_CONFIGS: ChartConfig[] = [
  {
    type: 'line',
    title: 'Trend Analysis',
    description: 'Track health metrics over time',
    icon: <TrendingUp className="h-5 w-5" />
  },
  {
    type: 'area',
    title: 'Wellness Overview',
    description: 'Visualize overall health patterns',
    icon: <Activity className="h-5 w-5" />
  },
  {
    type: 'bar',
    title: 'Metric Comparison',
    description: 'Compare different health factors',
    icon: <BarChart className="h-5 w-5" />
  },
  {
    type: 'radar',
    title: 'Health Radar',
    description: 'Multi-dimensional health assessment',
    icon: <Zap className="h-5 w-5" />
  },
  {
    type: 'composed',
    title: 'Comprehensive View',
    description: 'Combined health insights',
    icon: <Brain className="h-5 w-5" />
  }
];

const TIME_RANGES = [
  { value: '7d', label: '7 Days' },
  { value: '30d', label: '30 Days' },
  { value: '90d', label: '90 Days' },
  { value: '1y', label: '1 Year' }
];

export default function InteractiveHealthChart({
  data,
  riskLevel = 'normal',
  className = ''
}: InteractiveHealthChartProps) {
  const [selectedChart, setSelectedChart] = useState<ChartConfig>(CHART_CONFIGS[0]);
  const [timeRange, setTimeRange] = useState('30d');
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['sleep_hours', 'stress_level', 'mood_score']);

  // Filter data based on time range
  const filteredData = useMemo(() => {
    const now = new Date();
    const daysToSubtract = timeRange === '7d' ? 7 : timeRange === '30d' ? 30 : timeRange === '90d' ? 90 : 365;
    const cutoffDate = new Date(now.getTime() - (daysToSubtract * 24 * 60 * 60 * 1000));
    
    return data.filter(item => new Date(item.date) >= cutoffDate);
  }, [data, timeRange]);

  // Calculate health score
  const healthScore = useMemo(() => {
    if (filteredData.length === 0) return 0;
    
    const avgSleep = filteredData.reduce((sum, item) => sum + item.sleep_hours, 0) / filteredData.length;
    const avgStress = filteredData.reduce((sum, item) => sum + item.stress_level, 0) / filteredData.length;
    const avgExercise = filteredData.reduce((sum, item) => sum + item.exercise_frequency, 0) / filteredData.length;
    const avgMood = filteredData.reduce((sum, item) => sum + item.mood_score, 0) / filteredData.length;
    
    // Normalize scores (0-100)
    const sleepScore = Math.min(100, (avgSleep / 8) * 100);
    const stressScore = Math.max(0, 100 - (avgStress / 10) * 100);
    const exerciseScore = Math.min(100, (avgExercise / 7) * 100);
    const moodScore = (avgMood / 10) * 100;
    
    return Math.round((sleepScore + stressScore + exerciseScore + moodScore) / 4);
  }, [filteredData]);

  // Prepare radar chart data
  const radarData = useMemo(() => {
    if (filteredData.length === 0) return [];
    
    const avgData = {
      sleep: filteredData.reduce((sum, item) => sum + item.sleep_hours, 0) / filteredData.length,
      stress: filteredData.reduce((sum, item) => sum + item.stress_level, 0) / filteredData.length,
      exercise: filteredData.reduce((sum, item) => sum + item.exercise_frequency, 0) / filteredData.length,
      mood: filteredData.reduce((sum, item) => sum + item.mood_score, 0) / filteredData.length,
      energy: filteredData.reduce((sum, item) => sum + item.energy_level, 0) / filteredData.length,
      social: filteredData.reduce((sum, item) => sum + item.social_connections, 0) / filteredData.length
    };
    
    return [
      { metric: 'Sleep', value: avgData.sleep, fullMark: 10 },
      { metric: 'Stress', value: avgData.stress, fullMark: 10 },
      { metric: 'Exercise', value: avgData.exercise, fullMark: 7 },
      { metric: 'Mood', value: avgData.mood, fullMark: 10 },
      { metric: 'Energy', value: avgData.energy, fullMark: 10 },
      { metric: 'Social', value: avgData.social, fullMark: 10 }
    ];
  }, [filteredData]);

  // Prepare pie chart data
  const pieData = useMemo(() => {
    if (filteredData.length === 0) return [];
    
    const totalDays = filteredData.length;
    const goodDays = filteredData.filter(item => 
      item.sleep_hours >= 7 && 
      item.stress_level <= 5 && 
      item.mood_score >= 7
    ).length;
    
    const moderateDays = filteredData.filter(item => 
      (item.sleep_hours >= 6 && item.sleep_hours < 7) ||
      (item.stress_level > 5 && item.stress_level <= 7) ||
      (item.mood_score >= 5 && item.mood_score < 7)
    ).length;
    
    const challengingDays = totalDays - goodDays - moderateDays;
    
    return [
      { name: 'Good Days', value: goodDays, color: COLORS.low },
      { name: 'Moderate Days', value: moderateDays, color: COLORS.normal },
      { name: 'Challenging Days', value: challengingDays, color: COLORS.high }
    ];
  }, [filteredData]);

  const renderChart = () => {
    switch (selectedChart.type) {
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={filteredData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              {selectedMetrics.map((metric, index) => (
                <Line
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  stroke={Object.values(COLORS)[index % Object.values(COLORS).length]}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        );

      case 'area':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={filteredData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              {selectedMetrics.map((metric, index) => (
                <Area
                  key={metric}
                  type="monotone"
                  dataKey={metric}
                  fill={Object.values(COLORS)[index % Object.values(COLORS).length]}
                  stroke={Object.values(COLORS)[index % Object.values(COLORS).length]}
                  fillOpacity={0.3}
                />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        );

      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={filteredData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              {selectedMetrics.map((metric, index) => (
                <Bar
                  key={metric}
                  dataKey={metric}
                  fill={Object.values(COLORS)[index % Object.values(COLORS).length]}
                  radius={[4, 4, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        );

      case 'radar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="metric" />
              <PolarRadiusAxis angle={30} domain={[0, 10]} />
              <Radar
                name="Health Metrics"
                dataKey="value"
                stroke={COLORS.primary}
                fill={COLORS.primary}
                fillOpacity={0.3}
              />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        );

      case 'composed':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={filteredData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Bar yAxisId="left" dataKey="sleep_hours" fill={COLORS.primary} opacity={0.6} />
              <Line yAxisId="right" type="monotone" dataKey="stress_level" stroke={COLORS.high} strokeWidth={2} />
              <Line yAxisId="right" type="monotone" dataKey="mood_score" stroke={COLORS.low} strokeWidth={2} />
            </ComposedChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  };

  const getHealthScoreColor = (score: number) => {
    if (score >= 80) return COLORS.low;
    if (score >= 60) return COLORS.normal;
    return COLORS.high;
  };

  const getHealthScoreLabel = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Needs Attention';
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Health Score Overview */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-0 shadow-lg">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Overall Health Score</h3>
              <p className="text-sm text-gray-600">Based on your recent data</p>
            </div>
            <div className="text-right">
              <div 
                className="text-4xl font-bold"
                style={{ color: getHealthScoreColor(healthScore) }}
              >
                {healthScore}
              </div>
              <Badge 
                variant="secondary"
                style={{ backgroundColor: getHealthScoreColor(healthScore), color: 'white' }}
              >
                {getHealthScoreLabel(healthScore)}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Chart Controls */}
      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
            <div className="flex items-center gap-3">
              {selectedChart.icon}
              <div>
                <CardTitle className="text-lg">{selectedChart.title}</CardTitle>
                <p className="text-sm text-gray-600">{selectedChart.description}</p>
              </div>
            </div>
            <div className="flex gap-2">
              <Select value={timeRange} onValueChange={setTimeRange}>
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TIME_RANGES.map(range => (
                    <SelectItem key={range.value} value={range.value}>
                      {range.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Chart Type Selector */}
          <div className="flex flex-wrap gap-2 mb-6">
            {CHART_CONFIGS.map((config) => (
              <Button
                key={config.type}
                variant={selectedChart.type === config.type ? 'default' : 'outline'}
                size="sm"
                onClick={() => setSelectedChart(config)}
                className="flex items-center gap-2"
              >
                {config.icon}
                {config.title}
              </Button>
            ))}
          </div>

          {/* Metric Selector */}
          <div className="mb-6">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Select Metrics to Display</h4>
            <div className="flex flex-wrap gap-2">
              {[
                { key: 'sleep_hours', label: 'Sleep Hours', icon: <Calendar className="h-4 w-4" /> },
                { key: 'stress_level', label: 'Stress Level', icon: <Brain className="h-4 w-4" /> },
                { key: 'exercise_frequency', label: 'Exercise', icon: <Activity className="h-4 w-4" /> },
                { key: 'mood_score', label: 'Mood Score', icon: <Heart className="h-4 w-4" /> },
                { key: 'energy_level', label: 'Energy Level', icon: <Zap className="h-4 w-4" /> },
                { key: 'social_connections', label: 'Social Connections', icon: <Users className="h-4 w-4" /> }
              ].map((metric) => (
                <Button
                  key={metric.key}
                  variant={selectedMetrics.includes(metric.key) ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => {
                    if (selectedMetrics.includes(metric.key)) {
                      setSelectedMetrics(selectedMetrics.filter(m => m !== metric.key));
                    } else {
                      setSelectedMetrics([...selectedMetrics, metric.key]);
                    }
                  }}
                  className="flex items-center gap-2"
                >
                  {metric.icon}
                  {metric.label}
                </Button>
              ))}
            </div>
          </div>

          {/* Chart Display */}
          <div className="border rounded-lg p-4 bg-white">
            {filteredData.length === 0 ? (
              <div className="flex items-center justify-center h-80 text-gray-500">
                <div className="text-center">
                  <Calendar className="h-12 w-12 mx-auto mb-2 text-gray-300" />
                  <p>No data available for the selected time range</p>
                </div>
              </div>
            ) : (
              renderChart()
            )}
          </div>

          {/* Quick Stats */}
          {filteredData.length > 0 && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">
                  {filteredData.length}
                </div>
                <div className="text-sm text-gray-600">Data Points</div>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">
                  {Math.round(filteredData.reduce((sum, item) => sum + item.sleep_hours, 0) / filteredData.length * 10) / 10}
                </div>
                <div className="text-sm text-gray-600">Avg Sleep (hrs)</div>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-orange-600">
                  {Math.round(filteredData.reduce((sum, item) => sum + item.stress_level, 0) / filteredData.length * 10) / 10}
                </div>
                <div className="text-sm text-gray-600">Avg Stress</div>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">
                  {Math.round(filteredData.reduce((sum, item) => sum + item.mood_score, 0) / filteredData.length * 10) / 10}
                </div>
                <div className="text-sm text-gray-600">Avg Mood</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// Missing icon component
const Users = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
  </svg>
); 