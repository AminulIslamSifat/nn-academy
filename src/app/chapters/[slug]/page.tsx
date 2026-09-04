import { notFound } from 'next/navigation'
import { MDXRemote } from 'next-mdx-remote/rsc'
import { getChapterBySlug, getAllSlugs, getManifest } from '@/lib/mdx'
import { mdxComponents } from '@/components/mdx/registry'
import Sidebar from '@/components/Sidebar'
import ChapterHeader from '@/components/ChapterHeader'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import Link from 'next/link'
import type { Metadata } from 'next'

interface PageProps {
  params: { slug: string }
}

export function generateStaticParams() {
  return getAllSlugs().map((slug) => ({ slug }))
}

export function generateMetadata({ params }: PageProps): Metadata {
  const chapter = getChapterBySlug(params.slug)
  if (!chapter) return {}
  return {
    title: chapter.meta.title,
    description: chapter.meta.description,
  }
}

export default function ChapterPage({ params }: PageProps) {
  const chapter = getChapterBySlug(params.slug)
  if (!chapter) notFound()

  const manifest = getManifest()
  const { meta, content } = chapter

  // Find prev/next
  const allSlugs = getAllSlugs()
  const currentIdx = allSlugs.indexOf(meta.slug)
  const prevSlug = currentIdx > 0 ? allSlugs[currentIdx - 1] : null
  const nextSlug = currentIdx < allSlugs.length - 1 ? allSlugs[currentIdx + 1] : null

  return (
    <div className="flex min-h-screen">
      <Sidebar manifest={manifest} />

      {/* Main content area */}
      <main className="flex-1 lg:ml-72">
        <ChapterHeader
          title={meta.title}
          readTime={meta.readTime}
          codeTime={meta.codeTime}
        />

        {/* Chapter content */}
        <article className="mx-auto max-w-3xl px-6 py-12">
          <h1 className="text-3xl font-bold text-zinc-100 mb-4">{meta.title}</h1>
          {meta.description && (
            <p className="text-zinc-400 mb-8">{meta.description}</p>
          )}

          <div className="prose prose-invert prose-zinc max-w-none prose-headings:text-zinc-100 prose-a:text-accent-400 prose-strong:text-zinc-200">
            <MDXRemote source={content} components={mdxComponents} />
          </div>

          {/* Prev / Next navigation */}
          <nav className="mt-16 flex items-center justify-between border-t border-zinc-800 pt-8" aria-label="Chapter navigation">
            {prevSlug ? (
              <Link
                href={`/chapters/${prevSlug}`}
                className="flex items-center gap-2 text-sm text-zinc-400 hover:text-accent-400 transition"
              >
                <ChevronLeft size={16} />
                Previous
              </Link>
            ) : <div />}
            {nextSlug ? (
              <Link
                href={`/chapters/${nextSlug}`}
                className="flex items-center gap-2 text-sm text-zinc-400 hover:text-accent-400 transition"
              >
                Next
                <ChevronRight size={16} />
              </Link>
            ) : <div />}
          </nav>
        </article>
      </main>
    </div>
  )
}
