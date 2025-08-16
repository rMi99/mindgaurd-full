import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "MindGuard - Anonymous Mental Health Assessment",
  description: "AI-powered mental health risk assessment with cultural sensitivity",
  keywords: ["mental health", "assessment", "PHQ-9", "anonymous", "AI"],
  generator: "mindguard",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">{children}</div>
      </body>
    </html>
  )
}
