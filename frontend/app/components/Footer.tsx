import type { Language } from "@/lib/types"

interface FooterProps {
  language: Language
}

export default function Footer({ language }: FooterProps) {
  return (
    <footer className="bg-gray-50 border-t">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4">Emergency Resources</h3>
            <div className="space-y-2 text-sm text-gray-600">
              <p>Samaritans Lanka: 0717 171 171</p>
              <p>Sumithrayo: 0112 682 535</p>
              <p>Emergency Services: 1990</p>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4">Privacy & Security</h3>
            <div className="space-y-2 text-sm text-gray-600">
              <p>All data is anonymous</p>
              <p>No personal information stored</p>
              <p>GDPR compliant</p>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-4">Disclaimer</h3>
            <p className="text-sm text-gray-600">
              This tool is for screening purposes only and does not replace professional medical advice.
            </p>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t border-gray-200">
          <p className="text-center text-sm text-gray-500">
            © 2024 MindGuard. Designed for mental health awareness and support.
          </p>
        </div>
      </div>
    </footer>
  )
}
