import dynamic from 'next/dynamic'

// Dynamically import the dashboard to avoid SSR issues with webcam
const EnhancedHealthDashboard = dynamic(
  () => import('@/app/components/EnhancedHealthDashboard'),
  { 
    ssr: false,
    loading: () => (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-lg text-gray-600">Loading AI Health Dashboard...</p>
        </div>
      </div>
    )
  }
)

export default function HealthDashboardPage() {
  return <EnhancedHealthDashboard />
}
