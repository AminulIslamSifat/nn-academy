import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import ProgressLoader from '@/components/ProgressLoader'
import ServiceWorkerRegister from '@/components/ServiceWorkerRegister'
import '@/styles/globals.css'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
})

const jetbrains = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains',
})

export const metadata: Metadata = {
  title: {
    default: 'NumPy Neural Network Academy',
    template: '%s | NN Academy',
  },
  description: 'Learn deep learning from scratch — pure NumPy, zero frameworks. Interactive in-browser Python execution with visual tensor explorers.',
  icons: {
    icon: '/favicon.svg',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${jetbrains.variable} font-sans antialiased bg-surface-950 text-zinc-100`}>
        <ProgressLoader />
        <ServiceWorkerRegister />
        {children}
      </body>
    </html>
  )
}
