/** The Attune mark: the same drawing as the favicon (public/favicon.svg), so the tab and the page match. */
export function Logo({ className = 'h-7 w-7' }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="7" fill="#0f766e" />
      <path transform="translate(16 16.5) scale(1.22) translate(-16 -16)" fill="#fff"
        d="M16 7.5 23.6 24.5h-3.5l-1.5-3.6h-5.2l-1.5 3.6H8.4L16 7.5Zm0 6.3-1.6 4.1h3.2L16 13.8Z" />
    </svg>
  )
}
