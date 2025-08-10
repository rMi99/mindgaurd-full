import type { Language } from "@/lib/types"
import { getTranslation } from "@/lib/translations"
import { Button } from "@/components/ui/button"

interface HeaderProps {
  language: Language
}

export default function Header({ language }: HeaderProps) {
  return (
    <header className="bg-white shadow-sm border-b">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
              </div>
            </div>
            <div className="ml-4">
              <h1 className="text-2xl font-bold text-gray-900">{getTranslation(language, "title")}</h1>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <Button variant="ghost" asChild className="hidden sm:inline-flex">
              <a href="/research">Research</a>
            </Button>
            <Button variant="ghost" asChild className="hidden sm:inline-flex">
              <a href="/dashboard">Dashboard</a>
            </Button>
            <Button variant="ghost" asChild className="hidden sm:inline-flex text-red-600 hover:text-red-700">
              <a href="/admin">Admin</a>
            </Button>
            <div className="hidden sm:block">
              <span className="text-sm text-gray-500">Anonymous & Secure</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
